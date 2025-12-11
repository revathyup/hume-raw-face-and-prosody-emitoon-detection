from __future__ import annotations
import asyncio
import base64
import contextlib
import io
import os
from typing import Dict, Any, List, Optional

import numpy as np
import soundfile as sf
from openai import OpenAI
from furhat_realtime_api.async_furhat_client import AsyncFurhatClient
from furhat_realtime_api.events import Events
from config import settings
from emotion import EmotionHistory, build_emotion_context, describe_tone


class FurhatTextChat:
    """Furhat STT -> OpenAI text chat -> Furhat TTS + Hume PROSODY"""

    def __init__(
        self,
        furhat_client: AsyncFurhatClient,
        emotion_history: Optional[EmotionHistory] = None,
        prosody_sink: Optional[Any] = None,
    ):
        # --------------------------
        # OpenAI
        # --------------------------
        api_key_present = bool(os.getenv("OPENAI_API_KEY"))
        print(f"[INIT] OpenAI key present: {api_key_present} length={len(os.getenv('OPENAI_API_KEY') or '')}")
        self.client = OpenAI()
        self.model = settings.openai_chat_model
        self.emotion_history = emotion_history
        self.prosody_sink = prosody_sink

        # --------------------------
        # Furhat client
        # --------------------------
        self.furhat = furhat_client

        # Speech / ASR event handlers
        self.furhat.add_handler(Events.response_listen_start, self.on_listen_start)
        self.furhat.add_handler(Events.response_listen_end, self.on_listen_end)
        self.furhat.add_handler(Events.response_hear_start, self.on_hear_start)
        self.furhat.add_handler(Events.response_hear_end, self.on_hear_end)
        self.furhat.add_handler(Events.response_speak_end, self.on_speak_end)
        self.furhat.add_handler(Events.response_hear_partial, self.on_hear_partial)

        # --------------------------
        # PROSODY AUDIO STREAMING (delegated)
        # --------------------------
        audio_event = (
            getattr(Events, "response_audio_data", None)
            or getattr(Events, "response_audio", None)
        )
        if audio_event is not None:
            self.furhat.add_handler(audio_event, self.on_audio_frame)
        else:
            print("[WARN] Furhat SDK audio event not available; prosody disabled.")

        # --------------------------
        # State
        # --------------------------
        self.listening = False
        self.robot_speaking = False
        self.busy = False  # block GPT overlap

        # --------------------------
        # Conversation history
        # --------------------------
        self.history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a friendly social robot. "
                    "Keep responses short and spoken-friendly."
                )
            }
        ]

    # ----------------------------------------------------
    #   Furhat event handlers
    # ----------------------------------------------------

    async def on_listen_start(self, _data):
        self.listening = True
        print("[INFO] Listening started.")

    async def on_listen_end(self, data: Dict[str, Any]):
        cause = data.get("cause")
        print(f"[INFO] Listening ended. cause={cause}")
        self.listening = False
        await self.maybe_start_listening()

    async def on_hear_start(self, _data):
        print("[INFO] Hear start")

    async def on_hear_partial(self, data: Dict[str, Any]):
        text = data.get("text") or ""
        if text:
            print(f"[USER/PARTIAL]: {text}")

    async def on_hear_end(self, data: Dict[str, Any]):
        """Final ASR result."""
        text = data.get("text") or ""
        print("[USER]:", text)
        if self.prosody_sink and hasattr(self.prosody_sink, "set_current_text"):
            try:
                self.prosody_sink.set_current_text(text)
            except Exception:
                pass

        await self.stop_listening()

        if not text.strip():
            print("[INFO] Empty input. Restart listening.")
            await self.maybe_start_listening()
            return

        if self.busy:
            print("[WARN] Still processing previous message. Ignoring this one.")
            return

        self.busy = True

        # GPT reply
        try:
            reply = await self.get_openai_reply(text)
        except Exception as e:
            print(f"[ERROR] OpenAI failed: {e}")
            self.busy = False
            await self.maybe_start_listening()
            return

        print("[BOT]:", reply)

        # Speak via OpenAI TTS -> Furhat audio
        self.robot_speaking = True
        try:
            await self.speak_with_openai_tts(reply)
        except Exception as e:
            print(f"[ERROR] TTS failed: {e}")

        self.robot_speaking = False
        self.busy = False
        await self.maybe_start_listening()

    async def on_speak_end(self, _):
        print("[INFO] speak_end")
        self.robot_speaking = False
        self.busy = False
        await self.maybe_start_listening()

    # ----------------------------------------------------
    # PROSODY AUDIO HANDLING
    # ----------------------------------------------------

    def _extract_pcm_b64(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract base64 PCM payload from various Furhat event shapes."""
        if not data:
            return None
        debug_audio = os.environ.get("DEBUG_AUDIO_KEYS") == "1"
        if debug_audio:
            print(f"[DEBUG] audio payload keys: {list(data.keys())}")
        # Common Furhat shapes
        audio_obj = data.get("audio")
        if isinstance(audio_obj, dict):
            return (
                audio_obj.get("data")
                or audio_obj.get("buffer")
                or audio_obj.get("bytes")
            )
        mic_obj = data.get("microphone")
        if isinstance(mic_obj, dict):
            if debug_audio:
                print(f"[DEBUG] microphone keys: {list(mic_obj.keys())}")
                # Show length hints for common fields
                for k in ("data", "buffer", "bytes", "pcm", "pcm16"):
                    if k in mic_obj:
                        val = mic_obj[k]
                        try:
                            l = len(val)
                        except Exception:
                            l = "n/a"
                        print(f"[DEBUG] microphone[{k}] type={type(val)} len={l}")
            return (
                mic_obj.get("data")
                or mic_obj.get("buffer")
                or mic_obj.get("bytes")
                or mic_obj.get("pcm")
                or mic_obj.get("pcm16")
            )
        elif mic_obj is not None:
            if debug_audio:
                try:
                    l = len(mic_obj)  # type: ignore
                except Exception:
                    l = "n/a"
                print(f"[DEBUG] microphone value type={type(mic_obj)} len={l}")
            # If microphone payload is already bytes/bytearray, use directly
            return mic_obj
        # Some SDKs may put it at the top level
        return (
            data.get("data")
            or data.get("buffer")
            or data.get("bytes")
        )

    async def on_audio_frame(self, data):
        """Collect raw PCM from Furhat mic for Hume prosody."""
        try:
            # Ignore robot TTS audio to keep prosody tied to the user
            if self.robot_speaking:
                return
            if self.prosody_sink is None:
                return
            b64 = self._extract_pcm_b64(data)
            if not b64:
                return
            try:
                pcm = base64.b64decode(b64)
            except Exception:
                # If payload is already raw bytes string, fall back
                pcm = b64.encode("latin1") if isinstance(b64, str) else bytes(b64)
            if os.environ.get("DEBUG_AUDIO_KEYS") == "1":
                print(f"[DEBUG] audio b64 len={len(b64)} decoded bytes={len(pcm)}")
            self.prosody_sink.add_audio_chunk(pcm)
        except Exception as e:
            print("[ERROR] Audio frame decode failed:", e)

    # ----------------------------------------------------
    #   OpenAI integration
    # ----------------------------------------------------

    def _emotion_context_message(self) -> Optional[Dict[str, str]]:
        if not self.emotion_history:
            return None

        smoothed = self.emotion_history.smoothed()
        # Older versions of EmotionHistory don't expose a simple .trend();
        # use trend_bundle if available, otherwise fall back to a stable label.
        trend_bundle_fn = getattr(self.emotion_history, "trend_bundle", None)
        if callable(trend_bundle_fn):
            tb = trend_bundle_fn()
            trend = tb.get("overall", "stable")
        else:
            trend = "stable"
        tone = describe_tone(smoothed["valence"], smoothed["arousal"])
        context = build_emotion_context(
            smoothed["valence"],
            smoothed["arousal"],
            trend,
        )
        context += f"\nSuggested vocal tone: {tone}"
        return {"role": "system", "content": context}

    async def get_openai_reply(self, text: str) -> str:
        """Call OpenAI with optional emotion context injected."""
        messages: List[Dict[str, str]] = list(self.history)
        ctx = self._emotion_context_message()
        if ctx:
            messages.append(ctx)
        messages.append({"role": "user", "content": text})

        # Run blocking OpenAI call in thread to avoid blocking event loop
        resp = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        reply = resp.choices[0].message.content.strip()

        self.history.append({"role": "user", "content": text})
        self.history.append({"role": "assistant", "content": reply})
        # Trim history to avoid unbounded growth (keep system + last 8 exchanges)
        if len(self.history) > 18:
            # always keep the initial system prompt
            self.history = [self.history[0]] + self.history[-16:]

        return reply

    # ----------------------------------------------------
    #   Listen control + main loop
    # ----------------------------------------------------

    async def start_listening(self):
        if self.listening:
            return
        if hasattr(self.furhat, "request_listen_start"):
            try:
                await self.furhat.request_listen_start()
                self.listening = True
            except Exception as e:
                print(f"[WARN] request_listen_start failed: {e}")
        elif hasattr(self.furhat, "request_listen"):
            try:
                await self.furhat.request_listen()
                self.listening = True
            except Exception as e:
                print(f"[WARN] start_listening failed: {e}")
        else:
            # Older SDK: no explicit listen control; mark as listening to avoid loops
            self.listening = True

    async def stop_listening(self):
        if not self.listening:
            return
        try:
            if hasattr(self.furhat, "request_listen_stop"):
                await self.furhat.request_listen_stop()
            elif hasattr(self.furhat, "request_stop_listen"):
                await self.furhat.request_stop_listen()
        except Exception:
            pass
        self.listening = False

    async def maybe_start_listening(self):
        if self.busy or self.robot_speaking or self.listening:
            return
        await self.start_listening()

    async def run(self):
        """Entry point invoked by main.py."""
        print("FurhatTextChat.run started", flush=True)
        try:
            # Optional initial GPT greeting so the robot talks right away.
            # Controlled by INITIAL_GREETING env var (default: enabled).
            initial_greeting_enabled = os.environ.get("INITIAL_GREETING", "1") != "0"

            if initial_greeting_enabled:
                try:
                    # Small one-shot prompt to start the conversation.
                    prompt = (
                        "Greet the user briefly as a friendly social robot "
                        "and invite them to say something you'd like to talk about."
                    )
                    self.busy = True
                    self.robot_speaking = True
                    greeting = await self.get_openai_reply(prompt)
                    print("[BOT/INITIAL]:", greeting)
                    try:
                        await asyncio.wait_for(
                            self.furhat.request_speak_text(greeting, wait=True),
                            timeout=10.0,
                        )
                    except asyncio.TimeoutError:
                        print("[WARN] Initial TTS timeout — continuing.")
                    except Exception as e:
                        print(f"[ERROR] Initial TTS failed: {e}")
                except Exception as e:
                    print(f"[WARN] Initial OpenAI greeting failed: {e}")
                finally:
                    self.robot_speaking = False
                    self.busy = False

            # After any initial greeting, start listening for the user.
            await self.start_listening()

            # Keep the task alive; all work happens via event handlers
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise
        finally:
            await self.stop_listening()

    # ----------------------------------------------------
    #   OpenAI TTS -> Furhat audio helpers
    # ----------------------------------------------------

    async def speak_with_openai_tts(self, text: str) -> None:
        """
        Use OpenAI TTS to synthesize `text` to WAV, then stream PCM audio
        to Furhat via request_speak_audio_* so the robot speaks with GPT voice.
        """
        if not text or not text.strip():
            return

        def _tts_blocking() -> bytes:
            resp = self.client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=text,
                response_format="wav",
            )
            return resp.read()

        wav_bytes: bytes = await asyncio.to_thread(_tts_blocking)

        with io.BytesIO(wav_bytes) as buf:
            data, sample_rate = sf.read(buf, dtype="int16")

        if data.ndim == 2:
            data = data.mean(axis=1).astype("int16")

        pcm_bytes = data.tobytes()

        await self.furhat.request_speak_audio_start(
            sample_rate=int(sample_rate), lipsync=True
        )

        chunk_size = 3200
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i : i + chunk_size]
            if not chunk:
                continue
            b64 = base64.b64encode(chunk).decode("ascii")
            await self.furhat.request_speak_audio_data(b64)

        await self.furhat.request_speak_audio_end()

    # ----------------------------------------------------
    #   Overridden run() using OpenAI TTS
    # ----------------------------------------------------

    async def run(self):
        """Entry point invoked by main.py (OpenAI TTS version)."""
        print("FurhatTextChat.run started", flush=True)
        try:
            # Optional initial GPT greeting so the robot talks right away.
            initial_greeting_enabled = os.environ.get("INITIAL_GREETING", "1") != "0"

            if initial_greeting_enabled:
                try:
                    prompt = (
                        "Greet the user briefly as a friendly social robot "
                        "and invite them to say something you'd like to talk about."
                    )
                    self.busy = True
                    self.robot_speaking = True
                    greeting = await self.get_openai_reply(prompt)
                    print("[BOT/INITIAL]:", greeting)
                    try:
                        await self.speak_with_openai_tts(greeting)
                    except Exception as e:
                        print(f"[ERROR] Initial TTS failed: {e}")
                except Exception as e:
                    print(f"[WARN] Initial OpenAI greeting failed: {e}")
                finally:
                    self.robot_speaking = False
                    self.busy = False

            await self.start_listening()

            # Keep the task alive; all work happens via event handlers
            while True:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            raise
        finally:
            await self.stop_listening()

import asyncio
from typing import Dict, Any, List, Optional

from openai import OpenAI
from furhat_realtime_api.async_furhat_client import AsyncFurhatClient
from furhat_realtime_api.events import Events
from config import settings
from emotion import EmotionHistory, build_emotion_context, describe_tone


class FurhatTextChat:
    """Furhat STT -> OpenAI text chat -> Furhat TTS"""

    def __init__(self, furhat_client: AsyncFurhatClient, emotion_history: Optional[EmotionHistory] = None):
        # OpenAI
        self.client = OpenAI()
        self.model = settings.openai_chat_model
        self.emotion_history = emotion_history
        
        # Furhat (共享已有客户端，避免重复连接)
        self.furhat = furhat_client
        self.furhat.add_handler(Events.response_listen_start, self.on_listen_start)
        self.furhat.add_handler(Events.response_listen_end, self.on_listen_end)
        self.furhat.add_handler(Events.response_hear_start, self.on_hear_start)
        self.furhat.add_handler(Events.response_hear_end, self.on_hear_end)
        self.furhat.add_handler(Events.response_speak_end, self.on_speak_end)
        # 便于调试，记录 ASR partial
        self.furhat.add_handler(Events.response_hear_partial, self.on_hear_partial)

        # State
        self.listening = False
        self.robot_speaking = False
        self.busy = False     # Prevent overlapping OpenAI calls

        # conversation history
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
        # 自动尝试重新开始听
        await self.maybe_start_listening()

    async def on_hear_start(self, _data):
        print("[INFO] Hear start")

    async def on_hear_partial(self, data: Dict[str, Any]):
        text = data.get("text") or ""
        if text:
            print(f"[USER/PARTIAL]: {text}")

    async def on_hear_end(self, data: Dict[str, Any]):
        """Triggered when Furhat finishes ASR."""
        text = data.get("text") or ""
        print("[USER]:", text)

        # Stop listening during reply generation
        await self.stop_listening()

        # Skip empty speech
        if not text.strip():
            print("[INFO] Empty input. Restart listening.")
            await self.maybe_start_listening()
            return

        # Ignore if still busy
        if self.busy:
            print("[WARN] Still processing previous turn, ignoring input.")
            return

        self.busy = True

        # Generate reply
        try:
            reply = await self.get_openai_reply(text)
        except Exception as e:
            print(f"[ERROR] OpenAI call failed: {e}")
            self.busy = False
            await self.maybe_start_listening()
            return

        print("[BOT]:", reply)

        # 直接等待 TTS 结束，避免 speak_end 丢失导致一直 busy
        self.robot_speaking = True
        try:
            # 加超时守护，防止 Furhat 未返回导致卡在 busy/robot_speaking
            await asyncio.wait_for(
                self.furhat.request_speak_text(reply, wait=True),
                timeout=10.0,
            )
        except Exception as e:
            print(f"[ERROR] request_speak_text failed: {e}")
        except asyncio.TimeoutError:
            print("[WARN] TTS wait timed out; resetting state and continuing.")
        finally:
            self.robot_speaking = False
            self.busy = False
            await self.maybe_start_listening()

    async def on_speak_end(self, _):
        """After robot finishes speaking, resume listening."""
        print("[INFO] speak_end")
        self.robot_speaking = False
        self.busy = False
        await self.maybe_start_listening()

    # ----------------------------------------------------
    #   OpenAI
    # ----------------------------------------------------

    def _emotion_context_message(self) -> Optional[Dict[str, str]]:
        if not self.emotion_history:
            return None
        smoothed = self.emotion_history.smoothed()
        trend = self.emotion_history.trend()
        valence = smoothed.get("valence", 0.0)
        arousal = smoothed.get("arousal", 0.0)
        tone = describe_tone(valence, arousal)
        # Console visualization (non-blocking)
        print(
            f"[Emotion] V: {valence:+.2f}  A: {arousal:+.2f}  trend: {trend}",
            flush=True,
        )
        print(
            f"[GPT Influence] model adjusted tone: \"{tone}\"",
            flush=True,
        )
        content = build_emotion_context(valence, arousal, trend)
        return {"role": "system", "content": content}

    async def get_openai_reply(self, user_msg: str) -> str:
        self.history.append({"role": "user", "content": user_msg})

        loop = asyncio.get_running_loop()

        # Build prompt with optional emotion context (not persisted)
        emotion_context = self._emotion_context_message()
        if emotion_context:
            messages = self.history[:-1] + [emotion_context, self.history[-1]]
        else:
            messages = list(self.history)

        def _call():
            res = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return res.choices[0].message.content
        
        reply: str = await loop.run_in_executor(None, _call)
        self.history.append({"role": "assistant", "content": reply})
        return reply.strip()

    # ----------------------------------------------------
    #   Listening control
    # ----------------------------------------------------

    async def maybe_start_listening(self):
        """Only start listening if safe to do so."""
        if self.listening:
            return
        if self.robot_speaking:
            return
        if self.busy:
            return

        await self.start_listening()

    async def start_listening(self):
        try:
            await self.furhat.request_listen_start(
                partial=True,
                concat=True,
                stop_user_end=True,
                # 放宽静默/截断时间，避免用户说话稍慢就被切断
                no_speech_timeout=15.0,
                end_speech_timeout=1.8,
            )
            print("[INFO] Requesting listen...")
            # 如果 2 秒内没有收到 listen_start 事件，自动重试
            asyncio.create_task(self._ensure_listening_started())
        except Exception as e:
            self.listening = False
            print(f"[ERROR] request_listen_start failed: {e}")

    async def _ensure_listening_started(self):
        await asyncio.sleep(2.0)
        if not self.listening and not self.robot_speaking and not self.busy:
            print("[WARN] listen_start not confirmed, retrying...")
            await self.start_listening()

    async def stop_listening(self):
        if self.listening:
            try:
                await self.furhat.request_listen_stop()
            except Exception:
                pass
            self.listening = False

    # ----------------------------------------------------
    #   Entry point
    # ----------------------------------------------------

    async def run(self):
        if not getattr(self.furhat, "ws", None):
            print("Connecting to Furhat...")
            await self.furhat.connect()
            print("Connected! Starting listen...")
        else:
            print("Furhat already connected; starting listen...")
            print("Connected! Starting listen...")

        # 配置听写语言（如需改语言修改此处）
        try:
            await self.furhat.request_listen_config(languages=["en-US"])
        except Exception:
            pass

        # 可选：让 OpenAI 先打个招呼，再开始听
        try:
            if not self.robot_speaking and not self.busy:
                self.busy = True
                greet = await self.get_openai_reply("Say a short spoken-friendly greeting to start the conversation.")
                print("[BOT-GREETING]:", greet)
                self.robot_speaking = True
                try:
                    await self.furhat.request_speak_text(greet, wait=True)
                finally:
                    # 确保状态被清理，避免卡死
                    self.robot_speaking = False
                    self.busy = False
                    await self.maybe_start_listening()
        except Exception as e:
            print(f"[WARN] Greeting failed: {e}")
            self.busy = False
            self.robot_speaking = False

        await self.maybe_start_listening()

        # Keep the loop alive
        last_state_log = 0.0
        while True:
            await asyncio.sleep(0.1)
            # 每 2 秒打印一次状态，便于确认是否卡在 busy/robot_speaking
            now = asyncio.get_running_loop().time()
            if now - last_state_log >= 2.0:
                print(
                    f"[STATE] listening={self.listening} "
                    f"busy={self.busy} robot_speaking={self.robot_speaking}"
                )
                last_state_log = now

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from hume import AsyncHumeClient
from hume.expression_measurement.stream.stream import StreamModelPredictions
from hume.expression_measurement.stream import Config

from config import settings


class HumeProsodyStream:
    """Collect PCM 16 kHz mono chunks and send to Hume Prosody."""

    def __init__(self, min_bytes: Optional[int] = None, sample_rate: int = 16000):
        self.client = AsyncHumeClient(api_key=settings.hume_api_key)
        self.config = Config(prosody={})
        self.buffer = bytearray()
        self.sample_rate = sample_rate
        self.current_text: Optional[str] = None
        # Default to ~0.3s of audio at 16 kHz, 16-bit mono, unless overridden
        self.min_bytes = (
            min_bytes if min_bytes is not None else int(0.3 * sample_rate * 2)
        )
        self.max_buffer_bytes = int(3 * sample_rate * 2)  # cap to ~3s

    def add_audio_chunk(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        self.buffer.extend(pcm_bytes)
        if len(self.buffer) > self.max_buffer_bytes:
            self.buffer[:] = self.buffer[-self.max_buffer_bytes :]

    def set_current_text(self, text: str) -> None:
        text = (text or "").strip()
        self.current_text = text or None

    async def run(self, debug: bool = False):
        backoff = 1.0
        while True:
            try:
                if debug:
                    print("[PROSODY] Connecting to Hume...")
                async with self.client.expression_measurement.stream.connect() as socket:
                    if debug:
                        print("[PROSODY] Connected.")
                    backoff = 1.0

                    while True:
                        await asyncio.sleep(0.15)

                        if len(self.buffer) < self.min_bytes:
                            if debug:
                                print(
                                    f"[DEBUG] prosody buffer size={len(self.buffer)} "
                                    f"(need {self.min_bytes})"
                                )
                            continue

                        # send fixed-size chunks
                        while len(self.buffer) >= self.min_bytes:
                            pcm_bytes = bytes(self.buffer[: self.min_bytes])
                            del self.buffer[: self.min_bytes]
                            if debug:
                                print(f"[PROSODY] sending chunk bytes={len(pcm_bytes)}")

                            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
                            if pcm_array.size == 0:
                                continue

                            wav_buf = io.BytesIO()
                            sf.write(
                                wav_buf,
                                pcm_array,
                                self.sample_rate,
                                format="WAV",
                                subtype="PCM_16",
                            )
                            wav_bytes = wav_buf.getvalue()

                            # Use send_file (same pattern as HumeFaceStream)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                                tmp.write(wav_bytes)
                                tmp_path = Path(tmp.name)

                            try:
                                result = await socket.send_file(
                                    str(tmp_path),
                                    config=self.config,
                                )

                                if isinstance(result, StreamModelPredictions):
                                    if result.prosody and result.prosody.predictions:
                                        pred = result.prosody.predictions[-1]
                                        emotions = pred.emotions

                                        if self.current_text:
                                            print(f"[PROSODY TEXT] {self.current_text}")
                                        print("\n=== RAW PROSODY EMOTION SCORES ===")
                                        for e in emotions:
                                            print(f"{e.name:12s}: {e.score:.3f}")
                                        print("=================================\n")
                                    else:
                                        if debug:
                                            print(f"[PROSODY] no prosody predictions in result: {result}")
                                else:
                                    if debug:
                                        print(f"[PROSODY] unexpected result type: {type(result)} value={result}")
                            except Exception as e:
                                print("[PROSODY ERROR]", e)
                            finally:
                                tmp_path.unlink(missing_ok=True)
            except asyncio.CancelledError:
                # Allow task cancellation without noisy traceback
                return
            except Exception as e:
                print(f"[PROSODY] connection dropped, retrying in {backoff:.1f}s: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
# services/hume_prosody_raw.py

import asyncio
import base64
import io
import soundfile as sf

from hume import AsyncHumeClient
from hume.expression_measurement.stream.stream import StreamModelPredictions
from hume.expression_measurement.stream import Config


class HumeProsodyRaw:

    def __init__(self, api_key):
        self.api_key = api_key
        self.client = AsyncHumeClient(api_key)
        self.config = Config(prosody={})

        self.buffer = bytearray()
        self.sample_rate = 16000

    def add_audio_chunk(self, pcm_bytes):
        """Called by Furhat audio handler."""
        self.buffer.extend(pcm_bytes)

    def _pcm_to_wav(self, pcm):
        buf = io.BytesIO()
        sf.write(buf, pcm, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    async def run(self):
        """Continuously send buffered audio to Hume prosody and print raw results."""
        async with self.client.expression_measurement.stream.connect() as socket:
            print("[PROSODY] Connected to Hume.")

            while True:
                await asyncio.sleep(0.25)

                # require ~0.3 seconds of audio
                if len(self.buffer) < int(0.3 * self.sample_rate):
                    continue

                # copy + clear
                pcm = bytes(self.buffer)
                self.buffer.clear()

                wav_bytes = self._pcm_to_wav(pcm)

                result = await socket.send_bytes(wav_bytes, config=self.config)

                if isinstance(result, StreamModelPredictions) and result.prosody:
                    p = result.prosody.predictions[-1]
                    emb = p.emotions

                    print("\n=== RAW PROSODY EMOTION SCORES ===")
                    for e in emb:
                        print(f"{e.name:12s}: {e.score:.3f}")
                    print("=================================\n")

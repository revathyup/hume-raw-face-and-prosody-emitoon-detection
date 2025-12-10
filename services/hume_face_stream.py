import asyncio
import base64
import os
import tempfile
from pathlib import Path
from typing import Optional

from hume import AsyncHumeClient
from hume.expression_measurement.stream.stream.types import (
    Config,
    StreamFace,
    StreamModelPredictions,
    StreamErrorMessage,
    StreamWarningMessage,
)

from config import settings
from emotion import compute_va, EmotionHistory, describe_tone


class HumeFaceStream:
    """Continuously sends Furhat camera frames into Hume StreamFace model."""

    def __init__(self, history: Optional[EmotionHistory] = None):
        # 初始化 Hume Async Client
        self.client = AsyncHumeClient(api_key=settings.hume_api_key)
        # 配置 StreamFace 模型
        self.model_config = Config(face=StreamFace(), prosody={})
        # 控制打印频率（不降低检测频率，只减少终端刷屏）
        self.print_every = int(os.environ.get("HUME_PRINT_EVERY", "1"))
        self._frame_idx = 0  # 仅在拿到帧时计数，避免空循环干扰节流
        self._no_face_count = 0
        # 发送节流：每 N 帧发送一次，避免占满 Furhat WS 带宽导致语音指令超时
        self.send_every = int(os.environ.get("HUME_SEND_EVERY", "1"))
        # 共享情绪历史
        self.history = history

    async def run(self, camera):
        """Camera → Hume face recognition loop"""
        print("HumeFaceStream.run started", flush=True)
        print("Connecting to Hume Streaming API...", flush=True)

        # 启动摄像头流
        await camera.start()

        try:
            # 修复：connect() 不接受 options 参数
            async with self.client.expression_measurement.stream.connect() as socket:
                print("Hume face stream started.")

                while True:
                    # 从 Furhat camera 拿最新帧（非阻塞）
                    frame_b64 = await camera.get_frame()
                    if not frame_b64:
                        # 没有新帧，短暂等待
                        await asyncio.sleep(0.05)
                        continue

                    self._frame_idx += 1  # 仅在拿到帧时计数
                    should_send = (self._frame_idx % self.send_every == 0)
                    should_log = should_send or self.print_every == 1 or (
                        self._frame_idx % self.print_every == 0
                    )

                    # 节流：只每 send_every 帧发送一次，减轻 WS 压力
                    if not should_send:
                        continue
                    if should_log:
                        print(f"[HUME] got frame idx={self._frame_idx} len={len(frame_b64)}", flush=True)

                    # Base64 解码成 bytes
                    frame_bytes = base64.b64decode(frame_b64)

                    # Hume SDK 目前必须 send_file → 临时写成文件（Furhat 返回 base64 JPEG）
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(frame_bytes)
                        tmp_path = Path(tmp.name)

                    # 发送给 Hume - 修复：config 作为参数传递给 send_file
                    try:
                        if should_log:
                            print("[HUME] sending to hume...", flush=True)
                        result = await socket.send_file(
                            str(tmp_path),
                            config=self.model_config  # Config 在这里传递
                        )
                        if should_log:
                            print(f"[HUME] got hume result type={type(result)}", flush=True)
                    finally:
                        tmp_path.unlink(missing_ok=True)  # 删除临时文件

                    # 解析 Hume 返回 - 增加调试输出，便于定位没有情绪值的原因
                    if should_log:
                        print("[HUME] parsing result...", flush=True)
                    if isinstance(result, StreamModelPredictions):
                        if result.face and result.face.predictions:
                            preds = result.face.predictions
                            printed = False
                            for pred in preds:
                                if pred.emotions:
                                    printed = True
                                    # 将情绪列表转为 dict
                                    score_map = {
                                        e.name: e.score
                                        for e in pred.emotions
                                        if getattr(e, "name", None) and getattr(e, "score", None) is not None
                                    }
                                    va = compute_va(score_map)
                                    if self.history:
                                        self.history.add(va["valence"], va["arousal"], va["timestamp"])
                                        smoothed = self.history.smoothed()
                                        trend = self.history.trend()
                                        tone = describe_tone(smoothed["valence"], smoothed["arousal"])
                                        print(
                                            f"[Emotion] V: {smoothed['valence']:+.2f}  "
                                            f"A: {smoothed['arousal']:+.2f}  trend: {trend}",
                                            flush=True,
                                        )
                                        print(
                                            f"[GPT Influence] model adjusted tone: \"{tone}\"",
                                            flush=True,
                                        )
                                    if should_log:
                                        print("\n=== Hume Face Emotion Vector ===")
                                        for emotion in pred.emotions:
                                            if emotion.name and emotion.score is not None:
                                                print(f"{emotion.name}: {emotion.score:.3f}")
                                else:
                                    if should_log:
                                        print("Face detected but no emotions in prediction.")
                            if not printed:
                                # 如果有预测但未打印情绪，输出原始对象以便排查
                                if should_log:
                                    print(f"Face predictions present but no emotions: {preds}")
                        else:
                            self._no_face_count += 1
                            if should_log:
                                print(f"No face detected. Raw result: {result}")
                    elif isinstance(result, StreamErrorMessage):
                        print(f"Error from Hume API: {result}")
                    elif isinstance(result, StreamWarningMessage):
                        if should_log:
                            print(f"Warning from Hume API: {result}")
                    else:
                        if should_log:
                            print(f"Unexpected response type: {type(result)} ; value={result}")

                    await asyncio.sleep(0.15)
        finally:
            # 停止摄像头流
            await camera.stop()

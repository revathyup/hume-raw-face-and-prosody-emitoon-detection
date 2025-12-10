from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
from config import settings
from services.furhat_text_chat import FurhatTextChat
from services.furhat_camera import FurhatCamera
from services.hume_face_stream import HumeFaceStream
from furhat_realtime_api.async_furhat_client import AsyncFurhatClient
from emotion import EmotionHistory


async def main():
    enable_hume = os.environ.get("ENABLE_HUME", "1") != "0"

    # Shared emotion history for smoothing/trend across Hume and GPT
    emotion_history = EmotionHistory(maxlen=200)

    # Furhat client for chat/control
    furhat_ctrl = AsyncFurhatClient(settings.furhat_ip, auth_key=settings.furhat_auth_key)
    # Optional separate Furhat client for camera/Hume
    furhat_cam = AsyncFurhatClient(settings.furhat_ip, auth_key=settings.furhat_auth_key) if enable_hume else None

    chat = FurhatTextChat(furhat_ctrl, emotion_history=emotion_history)
    camera = FurhatCamera(furhat_cam) if enable_hume else None
    hume = HumeFaceStream(history=emotion_history) if enable_hume else None

    print("Connecting to Furhat (control)...")
    await furhat_ctrl.connect()
    print("Connected (control).")

    if enable_hume and furhat_cam is not None:
        print("Connecting to Furhat (camera)...")
        await furhat_cam.connect()
        print("Connected (camera).")

    await asyncio.sleep(1)

    # Two tasks running in parallel; tolerate Hume超时不影响对话
    tasks = [chat.run()]  # Furhat STT ↔ OpenAI Chat ↔ Furhat TTS
    if enable_hume and hume and camera:
        tasks.append(hume.run(camera))  # Camera → Hume → emotion vector

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                print(f"[WARN] background task error: {r}")
    finally:
        # Ensure resources are released even on Ctrl+C
        if camera:
            try:
                await camera.stop()
            except Exception:
                pass
        try:
            if enable_hume and furhat_cam and getattr(furhat_cam, "ws", None):
                await furhat_cam.ws.close()
        except Exception:
            pass
        try:
            if getattr(furhat_ctrl, "ws", None):
                await furhat_ctrl.ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())

from furhat_realtime_api.events import Events


class FurhatCamera:
    """Stream camera frames from Furhat and expose the latest frame."""

    def __init__(self, furhat_client):
        self.client = furhat_client
        self.latest_frame: str = ""
        # 订阅摄像头数据事件
        self.client.add_handler(Events.response_camera_data, self._on_camera_data)

    async def _on_camera_data(self, data):
        frame = data.get("png") or data.get("image")
        if frame:
            self.latest_frame = frame
        else:
            print(f"camera response missing png/image, keys={list(data.keys())}", flush=True)

    async def start(self):
        """Start streaming camera data."""
        try:
            await self.client.request_camera_start()
        except Exception as e:
            print(f"[WARN] camera start failed: {e}", flush=True)

    async def stop(self):
        """Stop streaming camera data."""
        try:
            await self.client.request_camera_stop()
        except Exception:
            pass

    async def get_frame(self):
        """Return latest frame (base64) or empty string if none yet."""
        return self.latest_frame or ""

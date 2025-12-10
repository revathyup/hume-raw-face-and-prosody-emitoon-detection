from dataclasses import dataclass
import os

@dataclass
class Settings:
    furhat_ip: str = "192.168.1.108"      # 你的 Furhat IP
    furhat_auth_key: str = os.getenv("FURHAT_AUTH_KEY", "")

    # OpenAI text model
    openai_chat_model: str = "gpt-4.1-mini"

    # Hume
    hume_api_key: str = os.getenv("HUME_API_KEY", "")

settings = Settings()

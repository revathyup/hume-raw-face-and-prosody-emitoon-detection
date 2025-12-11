import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Global configuration used across the system."""

    # Furhat
    furhat_ip: str = os.getenv("FURHAT_IP", "localhost")
    furhat_auth_key: str = os.getenv("FURHAT_AUTH_KEY", "")

    # Hume API key
    hume_api_key: str = os.getenv("HUME_API_KEY")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")

    # Sanity checks
    def validate(self):
        if not self.hume_api_key:
            raise RuntimeError("Missing HUME_API_KEY in .env")
        if not self.openai_api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in .env")


# Create global settings object
settings = Settings()
settings.validate()

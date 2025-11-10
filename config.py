import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

def _get_env_multi(*names, default=None):
    for n in names:
        v = os.getenv(n) or os.getenv(n.lower()) or os.getenv(n.upper())
        if v:
            # Allow users to put quotes in .env like openrouter_api_key="abc"
            return v.strip().strip(\"'\").strip('"')
    return default

class Config:
    # Accept OPENROUTER_API_KEY or openrouter_api_key
    OPENROUTER_API_KEY = _get_env_multi("OPENROUTER_API_KEY", "openrouter_api_key")
    # CrewAI uses OpenAI-compatible endpoints; OpenRouter is OpenAI-compatible
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")

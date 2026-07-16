"""Configuration for LG Custom RAG Agent."""
from dotenv import load_dotenv
import os
from functools import lru_cache

load_dotenv()


@lru_cache(maxsize=1)
def get_jina_api_key() -> str:
    """Get Jina AI API key from environment."""
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY environment variable not set")
    return api_key


@lru_cache(maxsize=1)
def get_zen_api_key() -> str:
    """Get Zen (Nemotron) API key from environment."""
    api_key = os.getenv("ZEN_API_KEY")
    if not api_key:
        raise ValueError("ZEN_API_KEY environment variable not set")
    return api_key


@lru_cache(maxsize=1)
def get_zen_api_base() -> str:
    """Get Zen API base URL."""
    return os.getenv("ZEN_API_BASE", "https://opencode.ai/zen/v1")


ZEN_MODEL = "nemotron-3-ultra-free"
"""Application settings loaded from environment variables.

All configuration is read from the environment with a SUPERBRAIN_ prefix,
or from a .env file. Settings are loaded once and cached via lru_cache.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration.

    All fields map to SUPERBRAIN_<FIELD_NAME> environment variables.
    """

    # Database
    database_url: str

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_qa_model: str = "llama3.1:8b"
    ollama_classification_model: str = "phi3:mini"
    ollama_digest_model: str = "llama3.1:8b"
    digest_schedule_hour: int = 7  # UTC hour to run daily digest

    # Crawler
    crawler_backend: Literal["spider", "httpx"] = "httpx"
    spider_api_key: str | None = None

    # Telegram
    telegram_bot_token: str | None = None
    telegram_webhook_url: str | None = None
    ngrok_authtoken: str | None = None

    # API (used by CLI)
    api_base_url: str = "http://localhost:8000"

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SUPERBRAIN_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()

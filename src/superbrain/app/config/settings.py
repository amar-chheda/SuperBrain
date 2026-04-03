"""Application settings definitions."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SUPERBRAIN_", extra="ignore")

    env: Literal["dev", "test", "prod"] = Field(default="dev")
    log_level: str = Field(default="INFO")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    database_url: str = Field(
        default="postgresql+psycopg://superbrain:superbrain@localhost:5432/superbrain"
    )

    telegram_bot_token: str | None = Field(default=None)

    model_runtime: Literal["local_stub", "ollama", "lmstudio"] = Field(default="local_stub")
    local_model_base_url: str = Field(default="http://localhost:11434")
    embedding_model_name: str = Field(default="nomic-embed-text")
    embedding_dimensions: int = Field(default=16)
    generation_model_name: str = Field(default="llama3.1:8b")
    model_request_timeout_seconds: float = Field(default=30.0)
    model_max_retries: int = Field(default=2)

    scheduler_enabled: bool = Field(default=False)
    scheduler_digest_cron: str = Field(default="0 7 * * *")
    metrics_backend: Literal["memory", "prometheus"] = Field(default="prometheus")

    feature_enable_digest: bool = Field(default=True)
    feature_enable_topic_classification: bool = Field(default=True)
    feature_enable_ingestion_api: bool = Field(default=True)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return validated and memoized application settings."""

    return AppSettings()

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
    ollama_query_analysis_model: str = "lfm2.5-thinking:1.2b"  # HyDE + intent extraction
    digest_schedule_hour: int = 7  # UTC hour to run daily digest
    ollama_keep_alive: str = "30m"  # keep hot-path models resident to avoid reload thrash

    # QA retrieval tuning (tune qa_min_* via the eval harness after re-embedding)
    qa_query_analysis_enabled: bool = True   # run lfm2 query analysis before retrieval
    qa_retrieval_top_k: int = 20             # candidates pulled per retrieval leg
    qa_evidence_top_n: int = 10              # fused chunks passed to the answer model
    qa_min_vector_similarity: float = 0.65   # T_v: nomic cosine floor (eval-tuned 2026-05-29)
    qa_min_bm25_score: float = 0.05          # T_b: normalized ts_rank_cd floor
    qa_url_max_chunks: int = 50              # chunks pulled when summarizing a URL directly

    # SLM reranker (precision gate over the recall pool)
    qa_rerank_enabled: bool = True
    ollama_rerank_model: str = "phi3:mini"   # fast pointwise relevance judge
    qa_rerank_pool_size: int = 12            # top fused candidates scored by the reranker
    qa_min_rerank_score: float = 0.5         # refuse if the best reranked score < this
    qa_rerank_keep_score: float = 0.5        # keep chunks scoring >= this as evidence

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

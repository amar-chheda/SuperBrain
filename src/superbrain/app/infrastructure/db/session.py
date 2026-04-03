"""Database engine and session factory helpers."""

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from superbrain.app.config.settings import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and cache SQLAlchemy engine from app settings."""

    settings = get_settings()
    engine_kwargs: dict[str, object] = {"pool_pre_ping": True}
    if settings.database_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(settings.database_url, **engine_kwargs)


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return a SQLAlchemy sessionmaker bound to the configured engine."""

    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)


def reset_db_caches() -> None:
    """Clear cached engine/session factories for tests and reconfiguration."""

    get_engine.cache_clear()
    get_session_factory.cache_clear()

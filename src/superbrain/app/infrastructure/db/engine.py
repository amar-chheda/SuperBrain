"""Async SQLAlchemy engine and session management.

The engine is initialised once at application startup and disposed on shutdown.
Never import _engine directly — use get_session() for all database access.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(database_url: str) -> None:
    """Initialise the async engine and session factory.

    Must be called once during application lifespan startup before any
    database operations are attempted.

    Args:
        database_url: asyncpg-compatible database URL.
    """
    global _engine, _session_factory
    _engine = create_async_engine(database_url, echo=False, pool_pre_ping=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def dispose_engine() -> None:
    """Dispose the engine and release all connections.

    Should be called during application lifespan shutdown.
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


def get_session_factory() -> "async_sessionmaker[AsyncSession] | None":
    """Return the current session factory, or None if not yet initialised.

    Use this instead of importing _session_factory directly — a direct import
    captures the None value at import time and never sees the live reference.

    Returns:
        The active session factory, or None before init_engine() is called.
    """
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for use as a FastAPI dependency.

    Yields:
        An open AsyncSession that is automatically closed after the request.

    Raises:
        RuntimeError: If the engine has not been initialised via init_engine().
    """
    if _session_factory is None:
        raise RuntimeError("Database engine not initialised. Call init_engine() first.")
    async with _session_factory() as session:
        yield session

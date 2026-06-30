"""Health check endpoint.

Returns the liveness status of the application and its critical dependencies:
the PostgreSQL database and the Ollama model runtime.
"""

import httpx
import structlog
from fastapi import APIRouter
from sqlalchemy import text

from superbrain.app.infrastructure.db.engine import get_session
from superbrain.settings import get_settings

router = APIRouter()
log = structlog.get_logger(__name__)


@router.get("/health")
async def health() -> dict:
    """Check application health and dependency connectivity.

    Probes the database with SELECT 1 and Ollama with GET /api/tags.
    Always returns HTTP 200 — callers must inspect the response body
    to determine whether dependencies are healthy.

    Returns:
        A dict with keys: status, db, ollama.
        status is always 'ok'. db and ollama are 'connected' or 'error'.
    """
    settings = get_settings()

    db_status = "connected"
    try:
        async for session in get_session():
            await session.execute(text("SELECT 1"))
    except Exception:
        log.warning("health.db_check_failed")
        db_status = "error"

    ollama_status = "connected"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
    except Exception:
        log.warning("health.ollama_check_failed")
        ollama_status = "error"

    return {"status": "ok", "db": db_status, "ollama": ollama_status}

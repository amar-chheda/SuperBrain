"""Telegram bot webhook handler.

Receives updates from Telegram and creates IngestionJob records for URLs.
The bot is disabled silently if SUPERBRAIN_TELEGRAM_BOT_TOKEN is not set.
"""

from datetime import UTC, datetime
from uuid import uuid4

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from superbrain.app.domain.entities import IngestionJob
from superbrain.app.infrastructure.db.engine import get_session
from superbrain.app.infrastructure.db.repositories.ingestion_job import (
    SqlAlchemyIngestionJobRepository,
)
from superbrain.settings import get_settings

router = APIRouter(prefix="/bot", tags=["bot"])
log = structlog.get_logger(__name__)


def _is_url(text: str) -> bool:
    """Return True if text looks like an https URL.

    Args:
        text: The string to check.

    Returns:
        True if the string starts with https://, False otherwise.
    """
    return text.strip().startswith("https://")


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request) -> JSONResponse:
    """Receive a Telegram update and handle it.

    Supported message formats:
    - A bare https URL → creates an IngestionJob with source='telegram'
    - /ask <question> → replies that QA is not yet available
    - Anything else → replies with usage instructions

    Returns:
        HTTP 200 always (Telegram requires a 200 response to stop retrying).
    """
    settings = get_settings()
    if not settings.telegram_bot_token:
        return JSONResponse(status_code=200, content={"ok": True})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=200, content={"ok": True})

    message = body.get("message") or body.get("edited_message")
    if not message:
        return JSONResponse(status_code=200, content={"ok": True})

    chat_id = message.get("chat", {}).get("id")
    text = (message.get("text") or "").strip()

    if not chat_id or not text:
        return JSONResponse(status_code=200, content={"ok": True})

    if _is_url(text):
        try:
            now = datetime.now(UTC)
            job = IngestionJob(
                id=uuid4(),
                input_type="url",
                input_value=text,
                status="pending",
                source="telegram",
                created_at=now,
                updated_at=now,
            )
            async for session in get_session():
                repo = SqlAlchemyIngestionJobRepository(session)
                await repo.save(job)

            log.info("bot.job_created", job_id=str(job.id), source="telegram")
            reply = f"Got it — ingesting {text}. Job ID: {job.id}"
        except Exception as exc:
            log.exception("bot.job_creation_failed", error=str(exc))
            reply = "Something went wrong — please try again."

    elif text.startswith("/ask"):
        reply = "QA is not yet available."

    else:
        reply = "Send me an https URL to ingest."

    await _send_reply(settings.telegram_bot_token, chat_id, reply)
    return JSONResponse(status_code=200, content={"ok": True})


async def _send_reply(token: str, chat_id: int, text: str) -> None:
    """Send a text reply to a Telegram chat.

    Args:
        token: The Telegram bot token.
        chat_id: The Telegram chat ID to reply to.
        text: The message text to send.
    """
    import httpx

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, json={"chat_id": chat_id, "text": text})
    except Exception as exc:
        log.warning("bot.reply_failed", error=str(exc))

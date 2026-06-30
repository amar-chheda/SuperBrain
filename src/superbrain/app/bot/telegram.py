"""Telegram bot webhook handler.

Receives updates from Telegram and creates IngestionJob records for URLs.
The bot is disabled silently if SUPERBRAIN_TELEGRAM_BOT_TOKEN is not set.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from superbrain.app.application.ingestion.use_case import IngestArticleUseCase
from superbrain.app.application.qa.use_case import AskQuestionUseCase
from superbrain.app.application.retrieval.vector_retriever import VectorRetriever
from superbrain.app.application.topics.use_cases import ClassifyArticleUseCase
from superbrain.app.domain.entities import IngestionJob
from superbrain.app.infrastructure.db.engine import get_session, get_session_factory
from superbrain.app.infrastructure.db.repositories.article_repo import (
    SqlAlchemyArticleRepository,
)
from superbrain.app.infrastructure.db.repositories.chunk_repo import (
    SqlAlchemyChunkRepository,
)
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import (
    ChunkRetrievalRepository,
)
from superbrain.app.infrastructure.db.repositories.ingestion_job import (
    SqlAlchemyIngestionJobRepository,
)
from superbrain.app.infrastructure.db.repositories.query_log_repo import (
    SqlAlchemyQueryLogRepository,
)
from superbrain.app.infrastructure.db.repositories.topic_repo import (
    SqlAlchemyArticleTopicMatchRepository,
    SqlAlchemyTopicRepository,
)
from superbrain.settings import get_settings

router = APIRouter(prefix="/bot", tags=["bot"])
log = structlog.get_logger(__name__)


def _is_url(text: str) -> bool:
    return text.strip().startswith("https://")


async def _telegram_ingest_background(
    job_id: UUID,
    chat_id: int,
    token: str,
    request: Request,
) -> None:
    """Run the full ingestion pipeline and notify the user on completion."""
    settings = get_settings()
    reply = "Something went wrong during ingestion — please try again."
    async for session in get_session():
        classify_use_case = None
        if getattr(request.app.state, "classification_enabled", False):
            classify_use_case = ClassifyArticleUseCase(
                article_repo=SqlAlchemyArticleRepository(session),
                topic_repo=SqlAlchemyTopicRepository(session),
                match_repo=SqlAlchemyArticleTopicMatchRepository(session),
                llm=request.app.state.llm_background,
                metrics=request.app.state.metrics,
                settings=settings,
            )
        use_case = IngestArticleUseCase(
            article_repo=SqlAlchemyArticleRepository(session),
            chunk_repo=SqlAlchemyChunkRepository(session),
            ingestion_job_repo=SqlAlchemyIngestionJobRepository(session),
            crawler=request.app.state.crawler,
            embedder=request.app.state.embedder,
            llm=request.app.state.llm_ingestion,
            chunker_factory=request.app.state.chunker_factory,
            metrics=request.app.state.metrics,
            settings=settings,
            classify_use_case=classify_use_case,
        )
        try:
            await use_case.execute(job_id)
            reply = "Done! Article ingested — you can now /ask questions about it."
        except Exception as exc:
            log.exception("bot.ingest_failed", job_id=str(job_id), error=str(exc))
            reply = f"Ingestion failed: {exc}"

    await _send_reply(token, chat_id, reply)


async def _telegram_qa_background(
    question: str,
    chat_id: int,
    token: str,
    request: Request,
) -> None:
    """Run the QA pipeline and send the answer back to the user."""
    settings = get_settings()
    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            chunk_repo = ChunkRetrievalRepository(session)
            use_case = AskQuestionUseCase(
                vector_retriever=VectorRetriever(
                    embedder=request.app.state.embedder,
                    chunk_repo=chunk_repo,
                ),
                llm=request.app.state.llm,
                query_log_repo=SqlAlchemyQueryLogRepository(session),
                metrics=request.app.state.metrics,
                settings=settings,
                article_repo=SqlAlchemyArticleRepository(session),
                chunk_repo=chunk_repo,
            )
            result = await use_case.execute(question)

        if result.aborted and result.abort_kind == "url_not_ingested":
            # The bot ingests a bare URL, so point the user straight at that path.
            reply = result.abort_reason or "I haven't ingested that article yet."
        elif result.aborted:
            async with session_factory() as topic_session:
                topic_repo = SqlAlchemyTopicRepository(topic_session)
                topics = await topic_repo.list_active()
            if topics:
                topic_list = "\n".join(f"• {t.name}" for t in topics)
                reply = f"I don't have enough information on that topic. Here's what I can talk about:\n\n{topic_list}"
            else:
                reply = "I don't have enough information on that topic in my knowledge base."
        else:
            reply = result.answer or "No answer generated."
            if result.citations:
                ref_lines = "\n".join(
                    f"[{c.number}] {c.article_url}" for c in result.citations
                )
                reply += f"\n\n{ref_lines}"
    except Exception as exc:
        log.exception("bot.qa_failed", error=str(exc))
        reply = f"QA failed: {exc}"

    await _send_reply(token, chat_id, reply)


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    """Receive a Telegram update and handle it.

    Supported message formats:
    - A bare https URL → creates an IngestionJob, runs pipeline, notifies on completion
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

            background_tasks.add_task(
                _telegram_ingest_background,
                job.id,
                chat_id,
                settings.telegram_bot_token,
                request,
            )
            log.info("bot.job_created", job_id=str(job.id), source="telegram")
            reply = f"Got it — ingesting {text}. I'll message you when it's done."
        except Exception as exc:
            log.exception("bot.job_creation_failed", error=str(exc))
            reply = "Something went wrong — please try again."

    elif text.startswith("/ask"):
        question = text[len("/ask"):].strip()
        if not question:
            reply = "Usage: /ask <your question>"
        else:
            background_tasks.add_task(
                _telegram_qa_background,
                question,
                chat_id,
                settings.telegram_bot_token,
                request,
            )
            reply = f'Thinking about "{question}"...'

    else:
        reply = "Send me an https URL to ingest."

    await _send_reply(settings.telegram_bot_token, chat_id, reply)
    return JSONResponse(status_code=200, content={"ok": True})


async def _send_reply(token: str, chat_id: int, text: str) -> None:
    import httpx

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(url, json={"chat_id": chat_id, "text": text})
    except Exception as exc:
        log.warning("bot.reply_failed", error=str(exc))

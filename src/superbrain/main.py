"""FastAPI application factory for Superbrain.

Creates and configures the FastAPI app with lifespan management,
middleware, route registration, and error handlers.
Import `app` to run with uvicorn.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
import nltk
import structlog
from fastapi import FastAPI

from superbrain.app.api.errors import register_error_handlers
from superbrain.app.api.router import api_router
from superbrain.app.application.digest.use_case import GenerateDailyDigestUseCase
from superbrain.app.application.metrics import InMemoryMetricsRecorder
from superbrain.app.application.scheduler.adapter import SchedulerAdapter
from superbrain.app.bot.telegram import router as bot_router
from superbrain.app.infrastructure.chunkers.factory import ChunkerFactory
from superbrain.app.infrastructure.tunnel import ngrok as ngrok_tunnel
from superbrain.app.infrastructure.crawlers.factory import get_crawler
from superbrain.app.infrastructure.db.engine import dispose_engine, get_session_factory, init_engine
from superbrain.app.infrastructure.db.repositories.article_repo import SqlAlchemyArticleRepository
from superbrain.app.infrastructure.db.repositories.digest_repo import SqlAlchemyDigestRepository
from superbrain.app.infrastructure.db.repositories.topic_repo import (
    SqlAlchemyArticleTopicMatchRepository,
    SqlAlchemyTopicRepository,
)
from superbrain.app.infrastructure.embeddings.ollama_embedder import OllamaEmbedder
from superbrain.app.infrastructure.llm.ollama_llm import OllamaLLM
from superbrain.logging_config import configure_logging
from superbrain.middleware import RequestIDMiddleware
from superbrain.settings import get_settings

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    Initialises the database engine, HTTP client, crawler, LLM, embedder,
    chunker factory, and metrics recorder on startup. Disposes all resources
    on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        Nothing — control passes to the running application.
    """
    settings = get_settings()
    init_engine(settings.database_url)

    # Download NLTK punkt tokenizer data if not already present
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    http_client = httpx.AsyncClient(
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        follow_redirects=True,
    )
    app.state.http_client = http_client
    app.state.crawler = get_crawler(settings, http_client)
    app.state.llm = OllamaLLM(settings=settings, http_client=http_client)
    app.state.embedder = OllamaEmbedder(settings=settings, http_client=http_client)
    app.state.chunker_factory = ChunkerFactory()
    app.state.metrics = InMemoryMetricsRecorder()
    # classify_use_case is assembled per-request in the background task
    # (needs a live DB session). Store a factory flag so ingestion knows it's enabled.
    app.state.classification_enabled = True

    # run_digest builds a fresh session per run — matches the API background task pattern.
    _llm = app.state.llm
    _metrics = app.state.metrics
    _settings = settings

    async def _run_digest(target_date, triggered_by):  # type: ignore[no-untyped-def]
        async with get_session_factory()() as session:
            use_case = GenerateDailyDigestUseCase(
                article_repo=SqlAlchemyArticleRepository(session),
                match_repo=SqlAlchemyArticleTopicMatchRepository(session),
                topic_repo=SqlAlchemyTopicRepository(session),
                digest_repo=SqlAlchemyDigestRepository(session),
                llm=_llm,
                metrics=_metrics,
                settings=_settings,
            )
            return await use_case.execute(target_date=target_date, triggered_by=triggered_by)

    scheduler = SchedulerAdapter(
        run_digest=_run_digest,
        schedule_hour=settings.digest_schedule_hour,
    )
    scheduler.start()
    app.state.scheduler = scheduler

    await ngrok_tunnel.start(settings)

    log.info(
        "superbrain.startup",
        database_url=settings.database_url.split("@")[-1],
        crawler_backend=settings.crawler_backend,
        embedding_model=settings.ollama_embedding_model,
        qa_model=settings.ollama_qa_model,
    )
    yield

    app.state.scheduler.stop()
    ngrok_tunnel.stop()
    await http_client.aclose()
    await dispose_engine()
    log.info("superbrain.shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Wires up logging, middleware, routers, and error handlers.
    Called once at import time.

    Returns:
        A fully configured FastAPI instance ready to serve requests.
    """
    settings = get_settings()
    configure_logging(settings)

    application = FastAPI(title="Superbrain", lifespan=lifespan)
    application.add_middleware(RequestIDMiddleware)
    application.include_router(api_router)
    application.include_router(bot_router)
    register_error_handlers(application)

    return application


app = create_app()

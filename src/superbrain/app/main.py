"""FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from superbrain.app.api.exception_handlers import register_exception_handlers
from superbrain.app.api.routes.digests import router as digests_router
from superbrain.app.api.routes.ingestion import router as ingestion_router
from superbrain.app.api.routes.qa import router as qa_router
from superbrain.app.api.routes.system import router as system_router
from superbrain.app.api.routes.topics import router as topics_router
from superbrain.app.config.settings import get_settings
from superbrain.app.observability.logging import configure_logging
from superbrain.app.observability.middleware import correlation_id_middleware


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize and tear down app-level resources."""

    settings = get_settings()
    configure_logging(settings.log_level)
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    app = FastAPI(title="Superbrain API", version="0.1.0", lifespan=lifespan)
    app.middleware("http")(correlation_id_middleware)
    register_exception_handlers(app)
    app.include_router(system_router)
    app.include_router(ingestion_router)
    app.include_router(digests_router)
    app.include_router(qa_router)
    app.include_router(topics_router)
    return app


app = create_app()

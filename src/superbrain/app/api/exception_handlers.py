"""Exception handler registration for FastAPI."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from superbrain.app.api.errors import AppError
from superbrain.app.errors import SuperbrainError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register application-specific and fallback exception handlers."""

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        """Render typed application errors in a consistent envelope."""

        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(SuperbrainError)
    async def handle_superbrain_error(_: Request, exc: SuperbrainError) -> JSONResponse:
        """Render typed domain/application errors consistently."""

        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": str(exc)}},
        )

    @app.exception_handler(Exception)
    async def handle_uncaught(_: Request, exc: Exception) -> JSONResponse:
        """Render uncaught exceptions as a generic internal error response."""

        logger.exception("uncaught_exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Unexpected server error."}},
        )

"""Global exception handlers for the FastAPI application.

Maps domain exceptions to structured HTTP error responses. Register all
handlers via register_error_handlers() in the app factory.
"""

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from superbrain.app.domain.exceptions import (
    ConflictError,
    CrawlerError,
    DomainValidationError,
    NotFoundError,
)

log = structlog.get_logger(__name__)


def _error_body(error_key: str, message: str, request: Request) -> dict:
    """Build the standard error response body.

    Args:
        error_key: Snake-case error code (e.g. 'not_found').
        message: Human-readable error description.
        request: The current request, used to extract request_id.

    Returns:
        Dict matching the standard error response shape.
    """
    request_id = getattr(request.state, "request_id", None)
    return {"error": error_key, "message": message, "request_id": request_id}


def register_error_handlers(app: object) -> None:
    """Register all domain exception handlers on the FastAPI app.

    Args:
        app: The FastAPI application instance.
    """
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_body("not_found", str(exc), request),
        )

    @app.exception_handler(ConflictError)
    async def conflict_handler(request: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=_error_body("conflict", str(exc), request),
        )

    @app.exception_handler(DomainValidationError)
    async def validation_handler(
        request: Request, exc: DomainValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_error_body("validation_error", str(exc), request),
        )

    @app.exception_handler(CrawlerError)
    async def crawler_error_handler(request: Request, exc: CrawlerError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_body("crawler_error", str(exc), request),
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_error_body("validation_error", str(exc), request),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_error_body("internal_error", "An unexpected error occurred", request),
        )

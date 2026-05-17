"""ASGI middleware for Superbrain.

RequestIDMiddleware assigns a UUID to every incoming request and binds it
into the structlog context so every log line emitted during request handling
automatically includes request_id.
"""

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects a per-request UUID into structlog context and response headers."""

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Assign a request_id, bind it to logs, and forward the request.

        Args:
            request: The incoming ASGI request.
            call_next: The next middleware or route handler.

        Returns:
            The response with X-Request-ID header set.
        """
        request_id = str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

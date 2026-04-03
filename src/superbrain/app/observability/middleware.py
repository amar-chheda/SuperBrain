"""Request correlation middleware."""

from fastapi import Request, Response
from starlette.middleware.base import RequestResponseEndpoint

from superbrain.app.observability.context import (
    generate_correlation_id,
    request_context,
)

REQUEST_ID_HEADER = "X-Request-ID"


async def correlation_id_middleware(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> Response:
    """Attach correlation IDs to request context and response headers."""

    request_id = request.headers.get(REQUEST_ID_HEADER, generate_correlation_id())
    with request_context(request_id):
        response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response

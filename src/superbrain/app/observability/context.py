"""Context helpers for request and job correlation IDs."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from uuid import uuid4

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)


def generate_correlation_id() -> str:
    """Generate a random correlation ID."""

    return uuid4().hex


def get_request_id() -> str | None:
    """Return the current request ID from context."""

    return _request_id_var.get()


def get_job_id() -> str | None:
    """Return the current job ID from context."""

    return _job_id_var.get()


def set_request_id(request_id: str) -> Token[str | None]:
    """Set the request ID in context and return its token."""

    return _request_id_var.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Reset request ID context using the provided token."""

    _request_id_var.reset(token)


def set_job_id(job_id: str) -> Token[str | None]:
    """Set the job ID in context and return its token."""

    return _job_id_var.set(job_id)


def reset_job_id(token: Token[str | None]) -> None:
    """Reset job ID context using the provided token."""

    _job_id_var.reset(token)


@contextmanager
def request_context(request_id: str) -> Iterator[None]:
    """Bind a request ID for the lifetime of the context manager."""

    token = set_request_id(request_id)
    try:
        yield
    finally:
        reset_request_id(token)


@contextmanager
def job_context(job_id: str) -> Iterator[None]:
    """Bind a job ID for the lifetime of the context manager."""

    token = set_job_id(job_id)
    try:
        yield
    finally:
        reset_job_id(token)

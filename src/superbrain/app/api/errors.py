"""API-facing error abstractions."""

from dataclasses import dataclass


@dataclass(slots=True)
class AppError(Exception):
    """Typed application error mapped to an HTTP response."""

    code: str
    message: str
    status_code: int = 400

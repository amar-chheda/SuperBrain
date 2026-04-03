"""Application-level exception taxonomy."""


class SuperbrainError(Exception):
    """Base application exception with code and HTTP status metadata."""

    code = "superbrain_error"
    status_code = 500


class ValidationError(SuperbrainError):
    """Raised when request or workflow inputs are invalid."""

    code = "validation_error"
    status_code = 400


class NotFoundError(SuperbrainError):
    """Raised when a requested resource is not found."""

    code = "not_found"
    status_code = 404


class ConflictError(SuperbrainError):
    """Raised when a resource state conflict occurs."""

    code = "conflict"
    status_code = 409

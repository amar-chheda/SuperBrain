"""Domain exception types for Superbrain.

These are the only exceptions that should cross the application boundary.
Infrastructure and application layers raise these; the API layer catches
them and maps them to the appropriate HTTP status codes.
"""


class NotFoundError(Exception):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource: str, identifier: str) -> None:
        """Initialise with a human-readable description of what was not found.

        Args:
            resource: The entity type (e.g. 'IngestionJob').
            identifier: The ID or key that was looked up.
        """
        self.resource = resource
        self.identifier = identifier
        super().__init__(f"{resource} {identifier} not found")


class ConflictError(Exception):
    """Raised when an operation would create a duplicate or conflicting state."""

    def __init__(self, message: str) -> None:
        """Initialise with a description of the conflict.

        Args:
            message: Human-readable conflict description.
        """
        super().__init__(message)


class DomainValidationError(Exception):
    """Raised when input fails domain-level business rule validation."""

    def __init__(self, message: str) -> None:
        """Initialise with a description of the validation failure.

        Args:
            message: Human-readable validation failure description.
        """
        super().__init__(message)


class CrawlerError(Exception):
    """Raised when a URL cannot be fetched or parsed by any crawler backend."""

    def __init__(self, url: str, reason: str, cause: Exception | None = None) -> None:
        """Initialise with the failed URL and a reason string.

        Args:
            url: The URL that failed to crawl.
            reason: Human-readable description of the failure.
            cause: The underlying exception, if any.
        """
        self.url = url
        self.reason = reason
        self.cause = cause
        super().__init__(f"Failed to crawl {url}: {reason}")


class LLMError(Exception):
    """Raised when an LLM call fails after all retries are exhausted."""

    def __init__(self, model: str, reason: str) -> None:
        """Initialise with the model name and failure reason.

        Args:
            model: The Ollama model tag that was called.
            reason: Human-readable description of the failure.
        """
        self.model = model
        self.reason = reason
        super().__init__(f"LLM call to {model} failed: {reason}")

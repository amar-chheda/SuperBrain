"""OpenTelemetry-compatible tracing hooks."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace


class TracingHook:
    """Minimal tracing helper used by application workflows."""

    def __init__(self, tracer_name: str = "superbrain") -> None:
        self._tracer = trace.get_tracer(tracer_name)

    @contextmanager
    def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> Iterator[Any]:
        """Create a span and optionally set attributes."""

        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

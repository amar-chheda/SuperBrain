"""Structured logging configuration using structlog.

Configures JSON output for production and human-readable text for development.
Every log line includes timestamp, level, and logger name. Request-scoped
context (request_id, job_id) is injected via structlog's context vars.
"""

import logging

import structlog

from superbrain.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog based on application settings.

    Sets up either JSON (production) or console (development) rendering.
    Must be called once at application startup before any log calls.

    Args:
        settings: Loaded application settings.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )

"""Structured logging bootstrap."""

import json
import logging
from datetime import UTC, datetime

from superbrain.app.observability.context import get_job_id, get_request_id


class JsonFormatter(logging.Formatter):
    """Serialize log records into a compact JSON payload."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON-serialized log entry for a record."""

        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
            "job_id": get_job_id(),
        }
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str) -> None:
    """Configure root logging to emit structured JSON logs."""

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)

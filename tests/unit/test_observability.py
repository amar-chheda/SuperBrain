"""Unit tests for observability helpers."""

import json
import logging

from superbrain.app.observability.context import (
    generate_correlation_id,
    get_job_id,
    get_request_id,
    job_context,
    request_context,
)
from superbrain.app.observability.logging import JsonFormatter
from superbrain.app.observability.timing import timed


def test_generate_correlation_id_is_non_empty() -> None:
    """Generated IDs should be non-empty strings."""

    value = generate_correlation_id()
    assert isinstance(value, str)
    assert value


def test_request_and_job_context_scoping() -> None:
    """Correlation IDs should be set only within context manager scope."""

    assert get_request_id() is None
    assert get_job_id() is None

    with request_context("req-1"):
        with job_context("job-1"):
            assert get_request_id() == "req-1"
            assert get_job_id() == "job-1"

    assert get_request_id() is None
    assert get_job_id() is None


def test_json_formatter_includes_correlation_fields() -> None:
    """Structured logs should include request and job IDs from context."""

    record = logging.LogRecord(
        name="superbrain.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="hello",
        args=(),
        exc_info=None,
    )

    with request_context("req-2"):
        with job_context("job-2"):
            payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "hello"
    assert payload["request_id"] == "req-2"
    assert payload["job_id"] == "job-2"


def test_timed_measures_elapsed_time() -> None:
    """Timing helper should populate elapsed milliseconds."""

    with timed() as result:
        _ = sum(range(1000))

    assert result.elapsed_ms >= 0

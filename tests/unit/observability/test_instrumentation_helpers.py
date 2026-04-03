"""Unit tests for observability instrumentation helpers."""

from datetime import UTC, datetime, timedelta

from superbrain.app.observability.metrics import InMemoryMetricsRecorder
from superbrain.app.observability.model_calls import ModelCallPayload


def test_in_memory_metrics_recorder_tracks_counters_and_observations() -> None:
    """Metrics recorder should capture increments and observations."""

    recorder = InMemoryMetricsRecorder()
    recorder.increment("ingestion.success_count")
    recorder.increment("ingestion.success_count", 2)
    recorder.observe("qa.answer_latency_ms", 35.5)

    assert recorder.counters["ingestion.success_count"] == 3
    assert recorder.observations["qa.answer_latency_ms"] == [35.5]


def test_model_call_payload_duration_ms() -> None:
    """Model call payload should compute duration in milliseconds."""

    started_at = datetime.now(UTC)
    finished_at = started_at + timedelta(milliseconds=123)

    payload = ModelCallPayload(
        provider="local",
        model_name="test",
        request_type="embed_query",
        prompt_template=None,
        started_at=started_at,
        finished_at=finished_at,
        status="success",
    )

    assert 122 <= payload.duration_ms <= 124

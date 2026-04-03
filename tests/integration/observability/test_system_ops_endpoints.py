"""Integration tests for model health and metrics endpoints."""

from fastapi.testclient import TestClient

from superbrain.app.main import create_app


def test_model_health_endpoint_returns_provider_statuses() -> None:
    """Model health endpoint should expose embedding and chat runtime states."""

    client = TestClient(create_app())
    response = client.get("/health/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["embedding_provider_healthy"] is True
    assert payload["chat_provider_healthy"] is True


def test_metrics_endpoint_exposes_text_payload() -> None:
    """Metrics endpoint should return a Prometheus text payload."""

    client = TestClient(create_app())
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

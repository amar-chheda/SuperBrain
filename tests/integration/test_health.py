"""Integration tests for system routes."""

from fastapi.testclient import TestClient

from superbrain.app.main import create_app
from superbrain.app.observability.middleware import REQUEST_ID_HEADER


def test_health_endpoint_returns_ok() -> None:
    """Health endpoint should return service metadata and OK status."""

    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "superbrain", "version": "0.1.0"}
    assert response.headers[REQUEST_ID_HEADER]


def test_health_echoes_request_id_header() -> None:
    """Middleware should preserve client-provided request IDs."""

    client = TestClient(create_app())
    response = client.get("/health", headers={REQUEST_ID_HEADER: "req-user-1"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "req-user-1"

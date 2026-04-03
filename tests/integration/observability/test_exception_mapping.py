"""Integration tests for exception taxonomy mapping."""

from fastapi import APIRouter
from fastapi.testclient import TestClient

from superbrain.app.errors import NotFoundError
from superbrain.app.main import create_app


def test_superbrain_not_found_error_maps_to_404() -> None:
    """NotFoundError should be mapped by exception handlers to HTTP 404."""

    app = create_app()
    router = APIRouter()

    @router.get("/test/not-found")
    def raise_not_found() -> None:
        raise NotFoundError("resource missing")

    app.include_router(router)

    client = TestClient(app)
    response = client.get("/test/not-found")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "not_found"


def test_validation_error_maps_to_400() -> None:
    """ValidationError from QA use case should map to HTTP 400."""

    client = TestClient(create_app())
    response = client.post("/qa/ask", json={"question": "   ", "top_k": 3})

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "validation_error"

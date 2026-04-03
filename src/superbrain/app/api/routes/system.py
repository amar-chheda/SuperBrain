"""System routes for service lifecycle and health checks."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from superbrain.app.api.dependencies import (
    get_chat_model_provider,
    get_embedding_provider,
    get_metrics_recorder,
)
from superbrain.app.application.ports import ChatModelProvider, EmbeddingProvider
from superbrain.app.observability.metrics import MetricsRecorder


class HealthResponse(BaseModel):
    """Response schema for the health endpoint."""

    status: str
    service: str
    version: str


class ModelHealthResponse(BaseModel):
    """Response schema for model runtime health checks."""

    embedding_provider_healthy: bool
    chat_provider_healthy: bool


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return health status for the Superbrain API."""

    return HealthResponse(status="ok", service="superbrain", version="0.1.0")


@router.get("/health/models", response_model=ModelHealthResponse)
def model_health(
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    chat_provider: Annotated[ChatModelProvider, Depends(get_chat_model_provider)],
) -> ModelHealthResponse:
    """Return health state for embedding and chat model providers."""

    return ModelHealthResponse(
        embedding_provider_healthy=embedding_provider.health_check(),
        chat_provider_healthy=chat_provider.health_check(),
    )


@router.get("/metrics")
def metrics(
    recorder: Annotated[MetricsRecorder, Depends(get_metrics_recorder)],
) -> Response:
    """Expose Prometheus metrics when backend supports rendering."""

    render = getattr(recorder, "render", None)
    if callable(render):
        return Response(content=render(), media_type="text/plain; version=0.0.4")
    return Response(content=b"", media_type="text/plain; version=0.0.4")

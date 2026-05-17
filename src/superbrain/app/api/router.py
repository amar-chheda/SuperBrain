"""Top-level API router that mounts all sub-routers.

Import and include api_router in the app factory to register all routes.
"""

from fastapi import APIRouter

from superbrain.app.api.digests import router as digests_router
from superbrain.app.api.health import router as health_router
from superbrain.app.api.ingestion import router as ingestion_router
from superbrain.app.api.observability import router as observability_router
from superbrain.app.api.qa import router as qa_router
from superbrain.app.api.topics import router as topics_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(ingestion_router)
api_router.include_router(qa_router)
api_router.include_router(topics_router)
api_router.include_router(digests_router)
api_router.include_router(observability_router)

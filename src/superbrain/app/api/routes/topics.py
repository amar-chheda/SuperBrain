"""Topic management and classification API routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from superbrain.app.api.dependencies import (
    get_classify_article_use_case,
    get_reclassify_articles_use_case,
    get_topic_service,
)
from superbrain.app.application.topics.classification import (
    ClassifyArticleUseCase,
    ReclassifyArticlesUseCase,
)
from superbrain.app.application.topics.models import TopicCreateInput, TopicUpdateInput
from superbrain.app.application.topics.service import TopicService
from superbrain.app.domain.models import TopicDefinition, TopicMatch, TopicVersion

router = APIRouter(prefix="/topics", tags=["topics"])


class TopicVersionResponse(BaseModel):
    """Serialized topic version payload."""

    id: UUID
    version: int
    description: str
    positive_examples: list[str]
    negative_examples: list[str]


class TopicResponse(BaseModel):
    """Serialized topic metadata and current version."""

    id: UUID
    name: str
    status: str
    priority: int
    current_version_id: UUID


class TopicWithVersionResponse(BaseModel):
    """Serialized topic with latest version details."""

    topic: TopicResponse
    latest_version: TopicVersionResponse


class CreateTopicRequest(BaseModel):
    """Request payload for topic creation."""

    name: str = Field(min_length=2, max_length=255)
    description: str = Field(min_length=5)
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    priority: int = Field(default=100, ge=1, le=1000)


class UpdateTopicRequest(BaseModel):
    """Request payload for topic update."""

    description: str = Field(min_length=5)
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    priority: int = Field(default=100, ge=1, le=1000)


class TopicMatchResponse(BaseModel):
    """Serialized article-topic match payload."""

    article_id: UUID
    topic_id: UUID
    topic_version_id: UUID
    score: float
    rationale: str
    disqualifiers: list[str]


class ReclassifyRequest(BaseModel):
    """Request payload for bulk reclassification."""

    article_ids: list[UUID] | None = None
    limit: int = Field(default=100, ge=1, le=1000)


class ReclassifyResponse(BaseModel):
    """Response payload for bulk reclassification workflow."""

    processed_articles: int


@router.post("", response_model=TopicWithVersionResponse)
def create_topic(
    payload: CreateTopicRequest,
    topic_service: Annotated[TopicService, Depends(get_topic_service)],
) -> TopicWithVersionResponse:
    """Create a new topic and return current version metadata."""

    topic = topic_service.create_topic(
        TopicCreateInput(
            name=payload.name,
            description=payload.description,
            positive_examples=tuple(payload.positive_examples),
            negative_examples=tuple(payload.negative_examples),
            priority=payload.priority,
        )
    )
    latest = topic_service.get_latest_version(topic.id)
    return TopicWithVersionResponse(
        topic=_to_topic_response(topic),
        latest_version=_to_version_response(latest),
    )


@router.put("/{topic_id}", response_model=TopicWithVersionResponse)
def update_topic(
    topic_id: UUID,
    payload: UpdateTopicRequest,
    topic_service: Annotated[TopicService, Depends(get_topic_service)],
) -> TopicWithVersionResponse:
    """Update topic definition by creating a new version."""

    topic = topic_service.update_topic(
        TopicUpdateInput(
            topic_id=topic_id,
            description=payload.description,
            positive_examples=tuple(payload.positive_examples),
            negative_examples=tuple(payload.negative_examples),
            priority=payload.priority,
        )
    )

    latest = topic_service.get_latest_version(topic.id)
    return TopicWithVersionResponse(
        topic=_to_topic_response(topic),
        latest_version=_to_version_response(latest),
    )


@router.post("/{topic_id}/deactivate", response_model=TopicResponse)
def deactivate_topic(
    topic_id: UUID,
    topic_service: Annotated[TopicService, Depends(get_topic_service)],
) -> TopicResponse:
    """Deactivate a topic."""

    topic = topic_service.deactivate_topic(topic_id)
    return _to_topic_response(topic)


@router.get("", response_model=list[TopicWithVersionResponse])
def list_topics(
    topic_service: Annotated[TopicService, Depends(get_topic_service)],
    active_only: bool = False,
) -> list[TopicWithVersionResponse]:
    """List topics with latest version payloads."""

    topics = topic_service.list_topics(active_only=active_only)
    output: list[TopicWithVersionResponse] = []
    for topic in topics:
        latest = topic_service.get_latest_version(topic.id)
        output.append(
            TopicWithVersionResponse(
                topic=_to_topic_response(topic),
                latest_version=_to_version_response(latest),
            )
        )
    return output


@router.post("/classify/articles/{article_id}", response_model=list[TopicMatchResponse])
def classify_article(
    article_id: UUID,
    classify_use_case: Annotated[
        ClassifyArticleUseCase,
        Depends(get_classify_article_use_case),
    ],
) -> list[TopicMatchResponse]:
    """Classify a single article against active topics."""

    matches = classify_use_case.classify(article_id)
    return [_to_match_response(match) for match in matches]


@router.post("/reclassify", response_model=ReclassifyResponse)
def reclassify_articles(
    payload: ReclassifyRequest,
    use_case: Annotated[ReclassifyArticlesUseCase, Depends(get_reclassify_articles_use_case)],
) -> ReclassifyResponse:
    """Trigger bulk reclassification for a scoped subset of articles."""

    processed = use_case.reclassify(article_ids=payload.article_ids, limit=payload.limit)
    return ReclassifyResponse(processed_articles=processed)


def _to_topic_response(topic: TopicDefinition) -> TopicResponse:
    return TopicResponse(
        id=topic.id,
        name=topic.name,
        status=topic.status.value,
        priority=topic.priority,
        current_version_id=topic.current_version_id,
    )


def _to_version_response(version: TopicVersion) -> TopicVersionResponse:
    return TopicVersionResponse(
        id=version.id,
        version=version.version,
        description=version.description,
        positive_examples=list(version.positive_examples),
        negative_examples=list(version.negative_examples),
    )


def _to_match_response(match: TopicMatch) -> TopicMatchResponse:
    return TopicMatchResponse(
        article_id=match.article_id,
        topic_id=match.topic_id,
        topic_version_id=match.topic_version_id,
        score=match.score,
        rationale=match.rationale,
        disqualifiers=list(match.disqualifiers),
    )

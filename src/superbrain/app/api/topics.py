"""Topic management and classification routes.

Provides full CRUD for topics (with versioned updates), manual reclassification
triggers, and article-topic match retrieval.

Routes:
    GET  /topics                          — list topics
    POST /topics                          — create topic
    GET  /topics/{id}                     — get topic
    PUT  /topics/{id}                     — versioned update (archives old, creates new)
    DELETE /topics/{id}                   — archive topic
    POST /topics/{id}/reclassify          — reclassify all articles against updated topic
    POST /topics/classify/articles/{id}   — classify a single article (manual trigger)
    GET  /articles/{id}/topics            — get all topic matches for an article
"""

from datetime import datetime
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from superbrain.app.application.topics.use_cases import (
    ClassifyArticleUseCase,
    ReclassifyTopicUseCase,
)
from superbrain.app.domain.entities import ArticleTopicMatch, Topic
from superbrain.app.domain.exceptions import NotFoundError
from superbrain.app.infrastructure.db.engine import get_session
from superbrain.app.infrastructure.db.repositories.article_repo import (
    SqlAlchemyArticleRepository,
)
from superbrain.app.infrastructure.db.repositories.topic_repo import (
    SqlAlchemyArticleTopicMatchRepository,
    SqlAlchemyTopicRepository,
)
from superbrain.settings import get_settings

router = APIRouter(tags=["topics"])
log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateTopicRequest(BaseModel):
    """Request body for creating a new topic."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1)
    examples: list[str] = Field(default_factory=list)
    priority: int = Field(default=0)


class UpdateTopicRequest(BaseModel):
    """Request body for updating a topic (creates a new version)."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1)
    examples: list[str] = Field(default_factory=list)
    priority: int = Field(default=0)


class TopicResponse(BaseModel):
    """API response shape for a topic."""

    id: UUID
    name: str
    version: int
    description: str
    examples: list[str]
    priority: int
    status: str

    model_config = {"from_attributes": True}


class ArticleTopicMatchResponse(BaseModel):
    """API response shape for an article-topic match."""

    id: UUID
    article_id: UUID
    topic_id: UUID
    topic_version: int
    confidence: str
    reason: str
    classified_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topic_to_response(topic: Topic) -> TopicResponse:
    """Convert a domain entity to the API response model.

    Args:
        topic: The domain entity.

    Returns:
        The serialisable response model.
    """
    return TopicResponse(
        id=topic.id,
        name=topic.name,
        version=topic.version,
        description=topic.description,
        examples=topic.examples,
        priority=topic.priority,
        status=topic.status,
    )


def _match_to_response(match: ArticleTopicMatch) -> ArticleTopicMatchResponse:
    """Convert a match domain entity to the API response model.

    Args:
        match: The domain entity.

    Returns:
        The serialisable response model.
    """
    return ArticleTopicMatchResponse(
        id=match.id,
        article_id=match.article_id,
        topic_id=match.topic_id,
        topic_version=match.topic_version,
        confidence=match.confidence,
        reason=match.reason,
        classified_at=match.classified_at,
    )


async def _reclassify_background(topic_id: UUID, request: Request) -> None:
    """Background task: reclassify all articles against the given topic.

    Args:
        topic_id: UUID of the topic to reclassify against.
        request: Original request, used to access app.state.
    """
    structlog.contextvars.bind_contextvars(topic_id=str(topic_id))
    async for session in get_session():
        settings = get_settings()
        use_case = ReclassifyTopicUseCase(
            article_repo=SqlAlchemyArticleRepository(session),
            topic_repo=SqlAlchemyTopicRepository(session),
            match_repo=SqlAlchemyArticleTopicMatchRepository(session),
            llm=request.app.state.llm_background,
            settings=settings,
        )
        await use_case.execute(topic_id)


# ---------------------------------------------------------------------------
# Routes — topics
# ---------------------------------------------------------------------------


@router.get("/topics", response_model=list[TopicResponse])
async def list_topics(
    include_archived: bool = False,
    session: AsyncSession = Depends(get_session),
) -> list[TopicResponse]:
    """List all topics.

    Args:
        include_archived: If True, include archived topics in the response.
        session: Injected async database session.

    Returns:
        List of topic response objects ordered by priority descending.
    """
    repo = SqlAlchemyTopicRepository(session)
    topics = await repo.list_all(include_archived=include_archived)
    return [_topic_to_response(t) for t in topics]


@router.post("/topics", status_code=status.HTTP_201_CREATED, response_model=TopicResponse)
async def create_topic(
    body: CreateTopicRequest,
    session: AsyncSession = Depends(get_session),
) -> TopicResponse:
    """Create a new topic.

    Args:
        body: Validated request body.
        session: Injected async database session.

    Returns:
        The created topic.
    """
    topic = Topic(
        id=uuid4(),
        name=body.name,
        version=1,
        description=body.description,
        examples=body.examples,
        priority=body.priority,
        status="active",
    )
    repo = SqlAlchemyTopicRepository(session)
    await repo.save(topic)
    log.info("topic.created", topic_id=str(topic.id), name=topic.name)
    return _topic_to_response(topic)


@router.get("/topics/{topic_id}", response_model=TopicResponse)
async def get_topic(
    topic_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> TopicResponse:
    """Retrieve a topic by ID.

    Args:
        topic_id: UUID of the topic.
        session: Injected async database session.

    Returns:
        The topic response.

    Raises:
        NotFoundError: If no topic with the given ID exists.
    """
    repo = SqlAlchemyTopicRepository(session)
    topic = await repo.find_by_id(topic_id)
    if topic is None:
        raise NotFoundError("Topic", str(topic_id))
    return _topic_to_response(topic)


@router.put("/topics/{topic_id}", response_model=TopicResponse)
async def update_topic(
    topic_id: UUID,
    body: UpdateTopicRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> TopicResponse:
    """Update a topic — archives the current version and creates a new one.

    The new version inherits the same UUID but increments the version counter.
    All articles are reclassified against the new version in the background.

    Args:
        topic_id: UUID of the topic to update.
        body: Validated request body with new field values.
        background_tasks: FastAPI background task queue.
        request: Current request (used to access app.state).
        session: Injected async database session.

    Returns:
        The new topic version.

    Raises:
        NotFoundError: If no topic with the given ID exists.
    """
    repo = SqlAlchemyTopicRepository(session)
    existing = await repo.find_by_id(topic_id)
    if existing is None:
        raise NotFoundError("Topic", str(topic_id))

    # Archive old version
    await repo.set_status(topic_id, "archived")

    # New version gets a new UUID — the DB PK is per-row, not per-logical-topic
    new_topic = Topic(
        id=uuid4(),
        name=body.name,
        version=existing.version + 1,
        description=body.description,
        examples=body.examples,
        priority=body.priority,
        status="active",
    )
    await repo.save(new_topic)
    log.info("topic.updated", old_id=str(topic_id),
             new_id=str(new_topic.id), new_version=new_topic.version)

    background_tasks.add_task(_reclassify_background, new_topic.id, request)

    return _topic_to_response(new_topic)


@router.delete("/topics/{topic_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive_topic(
    topic_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    """Archive a topic (soft delete — sets status to 'archived').

    Args:
        topic_id: UUID of the topic to archive.
        session: Injected async database session.

    Raises:
        NotFoundError: If no topic with the given ID exists.
    """
    repo = SqlAlchemyTopicRepository(session)
    topic = await repo.find_by_id(topic_id)
    if topic is None:
        raise NotFoundError("Topic", str(topic_id))
    await repo.set_status(topic_id, "archived")
    log.info("topic.archived", topic_id=str(topic_id))


@router.post("/topics/{topic_id}/reclassify", status_code=status.HTTP_202_ACCEPTED)
async def reclassify_topic(
    topic_id: UUID,
    background_tasks: BackgroundTasks,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict[str, str]:
    """Trigger reclassification of all articles against the given topic.

    Runs asynchronously in the background. Returns immediately.

    Args:
        topic_id: UUID of the topic to reclassify against.
        background_tasks: FastAPI background task queue.
        request: Current request (used to access app.state).
        session: Injected async database session.

    Returns:
        Accepted message with topic ID.

    Raises:
        NotFoundError: If no topic with the given ID exists.
    """
    repo = SqlAlchemyTopicRepository(session)
    topic = await repo.find_by_id(topic_id)
    if topic is None:
        raise NotFoundError("Topic", str(topic_id))

    background_tasks.add_task(_reclassify_background, topic_id, request)
    log.info("topic.reclassify_triggered", topic_id=str(topic_id))
    return {"detail": f"Reclassification queued for topic {topic_id}"}


# ---------------------------------------------------------------------------
# Routes — classification
# ---------------------------------------------------------------------------


@router.post(
    "/topics/classify/articles/{article_id}",
    response_model=list[ArticleTopicMatchResponse],
)
async def classify_article(
    article_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> list[ArticleTopicMatchResponse]:
    """Manually trigger classification of a single article against all active topics.

    Args:
        article_id: UUID of the article to classify.
        request: Current request (used to access app.state).
        session: Injected async database session.

    Returns:
        List of topic matches produced by the classifier.

    Raises:
        NotFoundError: If no article with the given ID exists.
    """
    settings = get_settings()
    use_case = ClassifyArticleUseCase(
        article_repo=SqlAlchemyArticleRepository(session),
        topic_repo=SqlAlchemyTopicRepository(session),
        match_repo=SqlAlchemyArticleTopicMatchRepository(session),
        llm=request.app.state.llm_background,
        metrics=request.app.state.metrics,
        settings=settings,
    )
    matches = await use_case.execute(article_id)
    return [_match_to_response(m) for m in matches]


@router.get(
    "/articles/{article_id}/topics",
    response_model=list[ArticleTopicMatchResponse],
)
async def get_article_topics(
    article_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> list[ArticleTopicMatchResponse]:
    """Return all topic matches for a given article.

    Args:
        article_id: UUID of the article.
        session: Injected async database session.

    Returns:
        All topic matches for the article, possibly empty.
    """
    repo = SqlAlchemyArticleTopicMatchRepository(session)
    matches = await repo.find_by_article(article_id)
    return [_match_to_response(m) for m in matches]

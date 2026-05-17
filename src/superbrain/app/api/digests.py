"""Digest routes — daily digest generation, listing, and retrieval."""

from datetime import date as DateType
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from pydantic import BaseModel

from superbrain.app.domain.entities import DigestItem, DigestRun
from superbrain.app.infrastructure.db.engine import get_session_factory
from superbrain.app.infrastructure.db.repositories.article_repo import (
    SqlAlchemyArticleRepository,
)
from superbrain.app.infrastructure.db.repositories.digest_repo import (
    SqlAlchemyDigestRepository,
)
from superbrain.app.infrastructure.db.repositories.topic_repo import (
    SqlAlchemyArticleTopicMatchRepository,
    SqlAlchemyTopicRepository,
)

router = APIRouter(prefix="/digests", tags=["digests"])


class TriggerRequest(BaseModel):
    date: DateType | None = None


class DigestItemResponse(BaseModel):
    topic_name: str
    summary: str
    article_titles: list[str]
    article_urls: list[str]
    position: int


class DigestRunResponse(BaseModel):
    id: UUID
    date_label: DateType
    status: str
    article_count: int
    section_count: int
    triggered_by: str
    started_at: str
    finished_at: str | None
    items: list[DigestItemResponse] = []


def _run_response(run: DigestRun, items: list[DigestItem] | None = None) -> DigestRunResponse:
    return DigestRunResponse(
        id=run.id,
        date_label=run.date_label,
        status=run.status,
        article_count=run.article_count,
        section_count=run.section_count,
        triggered_by=run.triggered_by,
        started_at=run.started_at.isoformat(),
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
        items=[
            DigestItemResponse(
                topic_name=i.topic_name,
                summary=i.summary,
                article_titles=i.article_titles,
                article_urls=i.article_urls,
                position=i.position,
            )
            for i in (items or [])
        ],
    )


async def _run_digest_background(target_date: DateType | None, request: Request) -> None:
    """Background task: build and run the digest use case with a fresh DB session."""
    from superbrain.app.application.digest.use_case import GenerateDailyDigestUseCase
    from superbrain.settings import get_settings

    settings = get_settings()
    async with get_session_factory()() as session:
        use_case = GenerateDailyDigestUseCase(
            article_repo=SqlAlchemyArticleRepository(session),
            match_repo=SqlAlchemyArticleTopicMatchRepository(session),
            topic_repo=SqlAlchemyTopicRepository(session),
            digest_repo=SqlAlchemyDigestRepository(session),
            llm=request.app.state.llm,
            metrics=request.app.state.metrics,
            settings=settings,
        )
        await use_case.execute(target_date=target_date, triggered_by="api")


@router.post("/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_digest(
    body: TriggerRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict:
    """Trigger a digest generation run.

    Returns immediately; generation continues in background.
    Optional body: {"date": "2024-01-15"} — defaults to yesterday if omitted.
    """
    background_tasks.add_task(_run_digest_background, body.date, request)
    return {
        "detail": "Digest generation queued",
        "date": str(body.date or "yesterday"),
    }


@router.get("", response_model=list[DigestRunResponse])
async def list_digests() -> list[DigestRunResponse]:
    """Return the 30 most recent digest runs, newest first."""
    async with get_session_factory()() as session:
        repo = SqlAlchemyDigestRepository(session)
        runs = await repo.list_runs(limit=30)
    return [_run_response(r) for r in runs]


@router.get("/{run_id}", response_model=DigestRunResponse)
async def get_digest(run_id: UUID) -> DigestRunResponse:
    """Return a digest run with its full list of topic sections."""
    async with get_session_factory()() as session:
        repo = SqlAlchemyDigestRepository(session)
        run = await repo.find_run_by_id(run_id)
        if not run:
            raise HTTPException(status_code=404, detail=f"Digest run {run_id} not found")
        items = await repo.find_items_by_run(run_id)
    return _run_response(run, items)

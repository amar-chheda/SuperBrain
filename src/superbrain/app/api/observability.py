"""Observability routes — metrics, model call logs, job traces, evals, query logs."""

from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from superbrain.app.application.evals.fixtures.qa_cases import QA_CASES
from superbrain.app.application.evals.fixtures.retrieval_cases import RETRIEVAL_CASES
from superbrain.app.application.evals.harness import run_all_evals
from superbrain.app.application.retrieval.vector_retriever import VectorRetriever
from superbrain.app.application.qa.use_case import AskQuestionUseCase
from superbrain.app.infrastructure.db.engine import get_session_factory
from superbrain.app.infrastructure.db.repositories.article_repo import SqlAlchemyArticleRepository
from superbrain.app.infrastructure.db.repositories.chunk_repo import SqlAlchemyChunkRepository
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import ChunkRetrievalRepository
from superbrain.app.infrastructure.db.repositories.ingestion_job import SqlAlchemyIngestionJobRepository
from superbrain.app.infrastructure.db.repositories.model_call_log_repo import SqlAlchemyModelCallLogRepository
from superbrain.app.infrastructure.db.repositories.query_log_repo import SqlAlchemyQueryLogRepository
from superbrain.app.infrastructure.db.repositories.topic_repo import SqlAlchemyArticleTopicMatchRepository
from superbrain.settings import get_settings

router = APIRouter(prefix="/observe", tags=["observability"])


@router.get("/metrics")
async def get_metrics(request: Request) -> dict:
    """Return a snapshot of all in-memory metrics counters and percentile summaries."""
    return request.app.state.metrics.snapshot()


@router.get("/model-calls")
async def list_model_calls(
    limit: int = Query(50, ge=1, le=500),
    request_type: str | None = Query(None),
    status: str | None = Query(None),
) -> list[dict]:
    """Return recent ModelCallLog records, filterable by request_type and status."""
    async with get_session_factory()() as session:
        repo = SqlAlchemyModelCallLogRepository(session)
        logs = await repo.list_recent(limit=limit, request_type=request_type, status=status)

    return [
        {
            "id": str(log.id),
            "provider": log.provider,
            "model_name": log.model_name,
            "request_type": log.request_type,
            "prompt_template": log.prompt_template,
            "duration_ms": log.duration_ms,
            "status": log.status,
            "retries": log.retries,
            "related_entity_id": str(log.related_entity_id) if log.related_entity_id else None,
            "started_at": log.started_at.isoformat(),
        }
        for log in logs
    ]


@router.get("/jobs/{job_id}/trace")
async def get_job_trace(job_id: UUID, request: Request) -> JSONResponse:
    """Full trace of a single ingestion job: article, chunks, model calls, topic matches."""
    async with get_session_factory()() as session:
        job_repo = SqlAlchemyIngestionJobRepository(session)
        job = await job_repo.find_by_id(job_id)
        if not job:
            return JSONResponse(status_code=404, content={"detail": f"Job {job_id} not found"})

        article_repo = SqlAlchemyArticleRepository(session)
        # Find article by canonical URL (ingestion uses canonicalised job input)
        from superbrain.app.infrastructure.crawlers.url_utils import canonicalise_url
        try:
            canonical = canonicalise_url(job.input_value)
        except Exception:
            canonical = job.input_value

        article = None
        all_active = await article_repo.list_all_active()
        for a in all_active:
            if a.canonical_url == canonical or a.url == job.input_value:
                article = a
                break

        model_calls: list = []
        chunk_count = 0
        strategy = None
        topic_matches: list = []

        if article:
            log_repo = SqlAlchemyModelCallLogRepository(session)
            model_calls_raw = await log_repo.list_by_entity(article.id)
            model_calls = [
                {
                    "request_type": c.request_type,
                    "model_name": c.model_name,
                    "duration_ms": c.duration_ms,
                    "status": c.status,
                    "prompt_template": c.prompt_template,
                }
                for c in model_calls_raw
            ]

            chunk_repo = SqlAlchemyChunkRepository(session)
            chunks = await chunk_repo.find_by_article(article.id)
            chunk_count = len(chunks)
            if chunks:
                strategy = chunks[0].strategy

            match_repo = SqlAlchemyArticleTopicMatchRepository(session)
            matches = await match_repo.find_by_article(article.id)
            topic_matches = [
                {"topic_id": str(m.topic_id), "confidence": m.confidence}
                for m in matches
            ]

    total_ms = sum(c["duration_ms"] for c in model_calls)

    return JSONResponse(content={
        "job": {
            "id": str(job.id),
            "status": job.status,
            "input_type": job.input_type,
            "input_value": job.input_value,
            "created_at": job.created_at.isoformat(),
        },
        "article": {
            "id": str(article.id) if article else None,
            "title": article.title if article else None,
            "chunk_count": chunk_count,
            "strategy": strategy,
        } if article else None,
        "model_calls": model_calls,
        "topic_matches": topic_matches,
        "total_duration_ms": total_ms,
    })


@router.get("/query-logs")
async def list_query_logs(
    limit: int = Query(20, ge=1, le=200),
) -> list[dict]:
    """Return recent QA query logs with latency and abort info."""
    async with get_session_factory()() as session:
        repo = SqlAlchemyQueryLogRepository(session)
        logs = await repo.list_recent(limit=limit)

    return [
        {
            "id": str(log.id),
            "question": log.question,
            "answer": log.answer,
            "aborted": log.aborted,
            "abort_reason": log.abort_reason,
            "retrieval_latency_ms": log.retrieval_latency_ms,
            "answer_latency_ms": log.answer_latency_ms,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


@router.get("/evals/run")
async def run_evals(request: Request) -> list[dict]:
    """Run the full eval suite and return pass/fail results.

    Expensive — makes real LLM calls. For demo and CI use only.
    """
    settings = get_settings()
    async with get_session_factory()() as session:
        chunk_repo = ChunkRetrievalRepository(session)
        query_log_repo = SqlAlchemyQueryLogRepository(session)

        vector_retriever = VectorRetriever(
            embedder=request.app.state.embedder,
            chunk_repo=chunk_repo,
        )
        qa_use_case = AskQuestionUseCase(
            vector_retriever=vector_retriever,
            llm=request.app.state.llm,
            query_log_repo=query_log_repo,
            metrics=request.app.state.metrics,
            settings=settings,
            article_repo=SqlAlchemyArticleRepository(session),
            chunk_repo=chunk_repo,
        )

        results = await run_all_evals(
            retrieval_cases=RETRIEVAL_CASES,
            qa_cases=QA_CASES,
            vector_retriever=vector_retriever,
            chunk_repo=chunk_repo,
            qa_use_case=qa_use_case,
        )

    return [
        {
            "name": r.name,
            "passed": r.passed,
            "score": r.score,
            "details": r.details,
            "duration_ms": r.duration_ms,
        }
        for r in results
    ]

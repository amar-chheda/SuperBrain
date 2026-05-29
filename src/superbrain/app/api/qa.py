"""QA routes — hybrid retrieval + grounded answer generation."""

from fastapi import APIRouter, Request, status
from pydantic import BaseModel

from superbrain.app.application.qa.use_case import AskQuestionUseCase, Citation
from superbrain.app.application.retrieval.bm25_retriever import BM25Retriever
from superbrain.app.application.retrieval.vector_retriever import VectorRetriever
from superbrain.app.infrastructure.db.engine import get_session_factory
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import (
    ChunkRetrievalRepository,
)
from superbrain.app.infrastructure.db.repositories.query_log_repo import (
    SqlAlchemyQueryLogRepository,
)
from superbrain.settings import get_settings

router = APIRouter(prefix="/qa", tags=["qa"])


class AskRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    number: int
    chunk_id: str
    article_title: str | None
    article_url: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str | None
    aborted: bool
    abort_reason: str | None = None
    citations: list[CitationResponse]
    retrieval_latency_ms: int = 0
    answer_latency_ms: int = 0


@router.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Submit a question to the hybrid retrieval + grounded QA pipeline.

    Returns a grounded answer with citations, or aborts with a reason if
    the knowledge base does not contain sufficient evidence to answer.
    """
    settings = get_settings()
    session_factory = get_session_factory()

    async with session_factory() as session:
        chunk_repo = ChunkRetrievalRepository(session)
        query_log_repo = SqlAlchemyQueryLogRepository(session)

        use_case = AskQuestionUseCase(
            vector_retriever=VectorRetriever(
                embedder=request.app.state.embedder,
                chunk_repo=chunk_repo,
            ),
            bm25_retriever=BM25Retriever(chunk_repo=chunk_repo),
            llm=request.app.state.llm,
            query_log_repo=query_log_repo,
            metrics=request.app.state.metrics,
            settings=settings,
        )

        result = await use_case.execute(body.question)

    return AskResponse(
        answer=result.answer,
        aborted=result.aborted,
        abort_reason=result.abort_reason,
        citations=[_citation_response(c) for c in result.citations],
        retrieval_latency_ms=result.retrieval_latency_ms,
        answer_latency_ms=result.answer_latency_ms,
    )


def _citation_response(c: Citation) -> CitationResponse:
    return CitationResponse(
        number=c.number,
        chunk_id=str(c.chunk_id),
        article_title=c.article_title,
        article_url=c.article_url,
        excerpt=c.excerpt,
    )

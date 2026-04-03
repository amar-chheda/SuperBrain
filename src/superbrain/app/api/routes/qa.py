"""Question-answering API routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from superbrain.app.api.dependencies import get_ask_question_use_case
from superbrain.app.application.qa.use_case import AskQuestionUseCase
from superbrain.app.domain.models import Citation

router = APIRouter(prefix="/qa", tags=["qa"])


class AskQuestionRequest(BaseModel):
    """Request payload for grounded question answering."""

    question: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)


class CitationResponse(BaseModel):
    """Citation payload for grounded answer response."""

    article_id: UUID
    article_title: str
    article_url: str
    chunk_id: UUID
    snippet: str
    rank: int
    score: float


class AskQuestionResponse(BaseModel):
    """Response payload for grounded answer endpoint."""

    request_id: UUID
    answer: str
    supported: bool
    citations: list[CitationResponse]


@router.post("/ask", response_model=AskQuestionResponse)
def ask_question(
    payload: AskQuestionRequest,
    use_case: Annotated[AskQuestionUseCase, Depends(get_ask_question_use_case)],
) -> AskQuestionResponse:
    """Answer user question using retrieved saved-article evidence."""

    result = use_case.ask(payload.question, top_k=payload.top_k)

    return AskQuestionResponse(
        request_id=result.request_id,
        answer=result.answer,
        supported=result.supported,
        citations=[_to_citation_response(citation) for citation in result.citations],
    )


def _to_citation_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        article_id=citation.article_id,
        article_title=citation.article_title,
        article_url=citation.article_url,
        chunk_id=citation.chunk_id,
        snippet=citation.snippet,
        rank=citation.rank,
        score=citation.score,
    )

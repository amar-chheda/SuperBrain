"""Structured QA models used for grounded answer generation."""

from pydantic import BaseModel, Field


class GeneratedAnswer(BaseModel):
    """Structured answer output from the chat model provider."""

    answer: str = Field(min_length=1)
    supported: bool
    citation_chunk_ids: list[str] = Field(default_factory=list)

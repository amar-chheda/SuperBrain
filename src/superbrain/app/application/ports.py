"""Application-level interfaces for external providers and core services."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from superbrain.app.application.qa.models import GeneratedAnswer
from superbrain.app.application.retrieval.models import EvidenceSet, RetrievalResult
from superbrain.app.domain.models import QueryResponse


@dataclass(slots=True, frozen=True)
class ExtractedArticle:
    """Structured extraction output used by ingestion workflows."""

    title: str
    canonical_url: str
    source_url: str
    domain: str
    author: str | None
    published_at: datetime | None
    body_text: str
    raw_html: str | None
    extraction_quality_score: float
    extraction_notes: str


@dataclass(slots=True, frozen=True)
class ChunkDraft:
    """Chunk draft before persistence."""

    index: int
    text: str
    token_count: int
    char_start: int
    char_end: int


class UrlCanonicalizer(Protocol):
    """Normalize URLs into canonical forms."""

    def canonicalize(self, url: str) -> str:
        """Return canonical representation for URL deduplication."""


class ArticleExtractor(Protocol):
    """Extract normalized article content from a source URL."""

    def extract(self, url: str) -> ExtractedArticle:
        """Extract canonical article content for a URL."""


class ChunkingStrategy(Protocol):
    """Split normalized article text into indexable chunks."""

    def chunk(self, text: str) -> list[ChunkDraft]:
        """Create chunk drafts from normalized text."""


class EmbeddingProvider(Protocol):
    """Generate vector embeddings for ingestion and retrieval."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return dense vectors for indexable texts."""

    def embed_query(self, text: str) -> list[float]:
        """Return dense vector for a query text."""

    def health_check(self) -> bool:
        """Return whether provider endpoint/model is reachable."""


class ChatModelProvider(Protocol):
    """Generate structured answers using provided evidence only."""

    def generate_answer(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer:
        """Return structured answer with citation chunk identifiers."""

    def health_check(self) -> bool:
        """Return whether provider endpoint/model is reachable."""


class RetrievalService(Protocol):
    """Retrieve candidate evidence snippets for user questions."""

    def retrieve(self, question: str, limit: int = 8) -> RetrievalResult:
        """Return ranked evidence relevant to a question."""


class Scheduler(Protocol):
    """Schedule recurring jobs."""

    def schedule(self, job_name: str, cron_expression: str) -> None:
        """Register or update a recurring job."""


class TelegramClient(Protocol):
    """Interact with Telegram for sending notifications."""

    def send_message(self, chat_id: str, text: str) -> None:
        """Send a message to a Telegram chat."""


class IngestionService(Protocol):
    """Orchestrate article ingestion workflow."""

    def ingest(self, url: str) -> str:
        """Ingest an article URL and return the created job ID."""


class QueryService(Protocol):
    """Answer user questions using grounded evidence."""

    def answer(self, question: str) -> QueryResponse:
        """Return grounded answer payload for a question."""


class DigestService(Protocol):
    """Build and dispatch digest messages."""

    def run_digest(self) -> int:
        """Run digest generation and return sent item count."""

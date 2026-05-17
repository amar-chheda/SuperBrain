"""Provider and service port interfaces for the application layer.

Defines the abstract contracts for all external capabilities the application
needs: crawling, embedding, LLM completion, and text chunking. Infrastructure
implementations live in infrastructure/ and are wired in at startup.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID


@dataclass
class CrawlResult:
    """The result of fetching and parsing a URL."""

    url: str
    canonical_url: str
    raw_text: str
    title: str | None
    author: str | None
    published_at: datetime | None
    status_code: int


class CrawlerPort(ABC):
    """Abstract interface for web page fetching and text extraction."""

    @abstractmethod
    async def fetch(self, url: str) -> CrawlResult:
        """Fetch a URL and extract its text content.

        Args:
            url: The URL to crawl.

        Returns:
            Parsed crawl result with extracted text and metadata.

        Raises:
            Exception: If the URL cannot be fetched or parsed.
        """


class EmbeddingPort(ABC):
    """Abstract interface for generating text embeddings."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed. Must be non-empty.

        Returns:
            List of embedding vectors, one per input text.
            Each vector has the same fixed dimension (768 for nomic-embed-text).
        """


class LLMPort(ABC):
    """Abstract interface for local LLM text completion."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = False,
        prompt_template: str = "unknown",
        related_entity_id: "UUID | None" = None,
    ) -> str:
        """Request a completion from a local LLM.

        Args:
            prompt: The full prompt string to send.
            model: The Ollama model tag to use (e.g. 'llama3.1:8b').
            json_mode: If True, instruct the model to respond with valid JSON.
            prompt_template: Name of the prompt template for audit logging.
            related_entity_id: UUID of the entity being processed, for audit logs.

        Returns:
            The model's completion text.

        Raises:
            LLMError: If the model call fails after retries.
        """


class ChunkerPort(ABC):
    """Abstract interface for splitting text into chunks."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        strategy: Literal["semantic", "recursive", "fixed"],
    ) -> list[str]:
        """Split text into chunks using the specified strategy.

        Args:
            text: The full text to split.
            strategy: Chunking algorithm to apply.

        Returns:
            Ordered list of text chunks. Each chunk is a non-empty string.
        """

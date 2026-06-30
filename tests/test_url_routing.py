"""URL-intent routing tests for AskQuestionUseCase — fakes only, no DB or Ollama.

Query analysis is disabled so URL detection is deterministic and the fake LLM is
only exercised for the grounded answer.
"""

from datetime import UTC, datetime
from uuid import uuid4

from superbrain.app.application.metrics import InMemoryMetricsRecorder
from superbrain.app.application.qa.use_case import AskQuestionUseCase
from superbrain.app.domain.entities import Article
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk
from superbrain.settings import Settings


class _FakeLLM:
    async def complete(self, prompt, *, model, json_mode=False,
                       prompt_template="unknown", related_entity_id=None):
        return "This article explains MCP. [1]\nSOURCES: 1"


class _FakeArticleRepo:
    def __init__(self, article):
        self._article = article

    async def find_by_canonical_url(self, canonical_url):
        return self._article


class _FakeChunkRepo:
    def __init__(self, chunks):
        self._chunks = chunks
        self.requested_article_id = None

    async def find_by_article(self, article_id, limit=50):
        self.requested_article_id = article_id
        return self._chunks


class _FakeQueryLogRepo:
    def __init__(self):
        self.saved = []

    async def save(self, log):
        self.saved.append(log)


def _settings():
    # Analysis off → URL detected deterministically; LLM only used for the answer.
    return Settings(
        database_url="postgresql+asyncpg://test/test",
        qa_query_analysis_enabled=False,
    )


def _article():
    return Article(
        id=uuid4(),
        url="https://example.com/a",
        canonical_url="https://example.com/a",
        content_hash="h",
        raw_text="...",
        title="MCP article",
        author=None,
        published_at=None,
        ingested_at=datetime.now(UTC),
        status="succeeded",
    )


def _chunk(article_id):
    return RankedChunk(
        id=uuid4(),
        article_id=article_id,
        content="MCP is a protocol for tool context.",
        chunk_index=0,
        title="MCP article",
        url="https://example.com/a",
        published_at=None,
        similarity_score=1.0,
    )


def _use_case(article, chunks):
    return AskQuestionUseCase(
        vector_retriever=object(),  # never used on the URL path
        llm=_FakeLLM(),
        query_log_repo=_FakeQueryLogRepo(),
        metrics=InMemoryMetricsRecorder(),
        settings=_settings(),
        article_repo=_FakeArticleRepo(article),
        chunk_repo=_FakeChunkRepo(chunks),
    )


async def test_url_hit_summarizes_that_article_only():
    art = _article()
    uc = _use_case(art, [_chunk(art.id)])
    result = await uc.execute("Tell me about this article: https://example.com/a")
    assert not result.aborted
    assert result.answer is not None
    assert result.citations
    assert result.citations[0].article_url == "https://example.com/a"


async def test_url_miss_refuses_with_ingest_hint():
    uc = _use_case(None, [])
    result = await uc.execute("Tell me about this article: https://example.com/missing")
    assert result.aborted
    assert result.abort_kind == "url_not_ingested"
    assert "ingest" in (result.abort_reason or "").lower()

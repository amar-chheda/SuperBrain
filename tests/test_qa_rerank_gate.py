"""A failed rerank must refuse safely, not fall through to unranked evidence.

Regression test for the bug where a reranker parse failure (fell_back=True)
left top_rerank as None, which skipped the refuse gate entirely and let the
answer model see plain RRF-ordered evidence — the weak signal reranking
exists to filter out.
"""

from uuid import uuid4

import superbrain.app.application.qa.use_case as use_case_module
from superbrain.app.application.metrics import InMemoryMetricsRecorder
from superbrain.app.application.qa.reranker import RerankResult
from superbrain.app.application.qa.use_case import AskQuestionUseCase
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk
from superbrain.settings import Settings


class _FakeLLM:
    async def complete(self, prompt, *, model, json_mode=False,
                       prompt_template="unknown", related_entity_id=None):
        raise AssertionError("answer model must not be called when rerank fails")


class _FakeVectorRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    async def retrieve_multi(self, probes, top_k):
        return [self._chunks]


class _FakeChunkRepo:
    async def find_by_text(self, keywords, top_k):
        return []


class _FakeQueryLogRepo:
    async def save(self, log):
        pass


def _chunk():
    return RankedChunk(
        id=uuid4(), article_id=uuid4(), content="relevant text", chunk_index=0,
        title="A", url="https://example.com/a", published_at=None, similarity_score=1.0,
    )


async def test_rerank_failure_refuses_instead_of_answering(monkeypatch):
    async def _failing_rerank(llm, *, model, query, chunks):
        return RerankResult(scores=[], fell_back=True)

    monkeypatch.setattr(use_case_module, "rerank", _failing_rerank)

    uc = AskQuestionUseCase(
        vector_retriever=_FakeVectorRetriever([_chunk(), _chunk(), _chunk()]),
        llm=_FakeLLM(),
        query_log_repo=_FakeQueryLogRepo(),
        metrics=InMemoryMetricsRecorder(),
        settings=Settings(
            database_url="postgresql+asyncpg://test/test",
            qa_query_analysis_enabled=False,
        ),
        article_repo=object(),  # never used on the topic-search path
        chunk_repo=_FakeChunkRepo(),
    )

    result = await uc.execute("how is qwen used for SQL queries?")

    assert result.aborted is True
    assert "reranker failed" in (result.abort_reason or "").lower()

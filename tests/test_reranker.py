"""Tests for the SLM reranker: score parsing, clamping, and graceful fallback — no Ollama."""

from uuid import uuid4

from superbrain.app.application.qa.reranker import rerank
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk


class _FakeLLM:
    def __init__(self, response: str = "", error: bool = False) -> None:
        self._response = response
        self._error = error

    async def complete(self, prompt, *, model, json_mode=False,
                       prompt_template="unknown", related_entity_id=None):
        if self._error:
            raise RuntimeError("ollama down")
        return self._response


def _chunk(content: str = "text") -> RankedChunk:
    return RankedChunk(
        id=uuid4(), article_id=uuid4(), content=content, chunk_index=0,
        title="A", url="https://x", published_at=None, similarity_score=0.5,
    )


async def test_rerank_happy_path_aligns_scores():
    llm = _FakeLLM('<think>weighing relevance</think>\n{"scores": [0.9, 0.2]}')
    res = await rerank(llm, model="phi3", query="q", chunks=[_chunk("a"), _chunk("b")])
    assert res.fell_back is False
    assert res.scores == [0.9, 0.2]


async def test_rerank_clamps_out_of_range_scores():
    llm = _FakeLLM('{"scores": [1.5, -0.2]}')
    res = await rerank(llm, model="phi3", query="q", chunks=[_chunk("a"), _chunk("b")])
    assert res.fell_back is False
    assert res.scores == [1.0, 0.0]


async def test_rerank_falls_back_on_score_count_mismatch():
    llm = _FakeLLM('{"scores": [0.9]}')  # only 1 score for 2 chunks
    res = await rerank(llm, model="phi3", query="q", chunks=[_chunk("a"), _chunk("b")])
    assert res.fell_back is True
    assert res.scores == []


async def test_rerank_empty_chunks_is_noop():
    res = await rerank(_FakeLLM(), model="phi3", query="q", chunks=[])
    assert res.scores == [] and res.fell_back is False


async def test_rerank_falls_back_on_llm_error():
    res = await rerank(_FakeLLM(error=True), model="phi3", query="q", chunks=[_chunk()])
    assert res.fell_back is True and res.scores == []


async def test_rerank_falls_back_on_unparseable_output():
    res = await rerank(_FakeLLM("not json at all"), model="phi3", query="q", chunks=[_chunk()])
    assert res.fell_back is True and res.scores == []

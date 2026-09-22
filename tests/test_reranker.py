"""Tests for the SLM reranker: pointwise scoring, clamping, and graceful fallback.

No Ollama — each test drives `rerank()` with a fake LLMPort standing in for the
per-passage calls it now makes.
"""

from uuid import uuid4

from superbrain.app.application.qa.reranker import rerank
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import RankedChunk


class _FakeLLM:
    """Fake LLMPort.

    `responses`, when given, overrides `response`/`error`: one entry per
    expected call, consumed in order — a None entry raises for that call,
    modelling one chunk's scoring call failing while others succeed.
    """

    def __init__(self, response: str = "", error: bool = False,
                 responses: list[str | None] | None = None) -> None:
        self._response = response
        self._error = error
        self._responses = responses
        self._call = 0

    async def complete(self, prompt, *, model, json_mode=False,
                       prompt_template="unknown", related_entity_id=None):
        if self._responses is not None:
            item = self._responses[self._call]
            self._call += 1
            if item is None:
                raise RuntimeError("ollama down")
            return item
        if self._error:
            raise RuntimeError("ollama down")
        return self._response


def _chunk(content: str = "text") -> RankedChunk:
    return RankedChunk(
        id=uuid4(), article_id=uuid4(), content=content, chunk_index=0,
        title="A", url="https://x", published_at=None, similarity_score=0.5,
    )


async def test_rerank_scores_each_passage_independently():
    llm = _FakeLLM(responses=['{"score": 0.9}', '{"score": 0.2}'])
    res = await rerank(llm, model="phi3", query="q", chunks=[_chunk("a"), _chunk("b")])
    assert res.fell_back is False
    assert res.scores == [0.9, 0.2]


async def test_rerank_clamps_out_of_range_scores():
    llm = _FakeLLM(responses=['{"score": 1.5}', '{"score": -0.2}'])
    res = await rerank(llm, model="phi3", query="q", chunks=[_chunk("a"), _chunk("b")])
    assert res.fell_back is False
    assert res.scores == [1.0, 0.0]


async def test_rerank_one_failed_chunk_defaults_to_zero_not_full_fallback():
    """The bug this replaces: a batched call losing count of one item used to
    discard every score. Pointwise isolates the failure to that one chunk."""
    llm = _FakeLLM(responses=['{"score": 0.8}', None, '{"score": 0.6}'])
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    res = await rerank(llm, model="phi3", query="q", chunks=chunks)
    assert res.fell_back is False
    assert res.scores == [0.8, 0.0, 0.6]


async def test_rerank_empty_chunks_is_noop():
    res = await rerank(_FakeLLM(), model="phi3", query="q", chunks=[])
    assert res.scores == [] and res.fell_back is False


async def test_rerank_falls_back_only_when_every_call_fails():
    res = await rerank(_FakeLLM(error=True), model="phi3", query="q",
                        chunks=[_chunk("a"), _chunk("b")])
    assert res.fell_back is True and res.scores == []


async def test_rerank_falls_back_when_every_call_is_unparseable():
    res = await rerank(_FakeLLM("not json at all"), model="phi3", query="q", chunks=[_chunk()])
    assert res.fell_back is True and res.scores == []

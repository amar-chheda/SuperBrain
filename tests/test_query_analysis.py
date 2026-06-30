"""Tests for query decomposition: extraction, directive routing, fallback — no Ollama."""

from superbrain.app.application.qa.query_analysis import (
    _deterministic_split,
    _extract_json,
    analyze_query,
    detect_url,
    raw_analysis,
)


class _FakeLLM:
    """Minimal LLMPort stand-in returning a canned response or raising."""

    def __init__(self, response: str = "", error: bool = False) -> None:
        self._response = response
        self._error = error

    async def complete(self, prompt, *, model, json_mode=False,
                       prompt_template="unknown", related_entity_id=None):
        if self._error:
            raise RuntimeError("ollama down")
        return self._response


def test_detect_url_trims_trailing_punctuation():
    assert detect_url("see https://example.com/a.html.") == "https://example.com/a.html"
    assert detect_url("no url here") is None


def test_extract_json_strips_think_block():
    raw = "<think>let me reason about this</think>\n{\"search_query\": \"x\", \"url\": null}"
    assert _extract_json(raw) == {"search_query": "x", "url": None}


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("there is no json here") is None


def test_deterministic_split_separates_topic_from_directives():
    topic, directives = _deterministic_split(
        "tell me more about simulated society — be detailed and use bullet points"
    )
    assert "simulated society" in topic
    assert "tell me more about" not in topic
    assert "bullet points" not in topic  # directive removed from the topic
    assert "detailed" in directives and "bullet points" in directives


def test_raw_analysis_routes_directives_and_detects_url():
    a = raw_analysis("explain https://x.com/a in bullet points")
    assert a.intent == "summarize_url"
    assert a.url == "https://x.com/a"
    assert "bullet points" in a.answer_directives
    assert a.fell_back is False


async def test_analyze_query_happy_path_uses_search_query_and_directives():
    llm = _FakeLLM(
        '<think>reasoning</think>\n'
        '{"search_query": "simulated society experiment", '
        '"keywords": "simulated society experiment", '
        '"answer_directives": "be detailed; use bullet points", '
        '"hypothetical_passage": "A simulated society runs many agents.", '
        '"intent": "summarize_topic", "url": null}'
    )
    q = "tell me more about the simulated society experiment — be detailed and use bullet points"
    result = await analyze_query(llm, model="lfm2", question=q)
    assert result.search_query == "simulated society experiment"
    assert "bullet points" in result.answer_directives
    assert result.intent == "summarize_topic"
    assert result.fell_back is False


async def test_analyze_query_rejects_hallucinated_search_query():
    # Model returns a topic with words absent from the question → use deterministic strip.
    llm = _FakeLLM(
        '{"search_query": "quantum chromodynamics lattice", '
        '"keywords": "x", "intent": "summarize_topic", "url": null}'
    )
    result = await analyze_query(
        llm, model="lfm2", question="tell me about model context protocol"
    )
    assert result.search_query == "model context protocol"
    assert result.fell_back is False  # the call succeeded; only the topic was overridden


async def test_analyze_query_falls_back_on_llm_error():
    llm = _FakeLLM(error=True)
    result = await analyze_query(llm, model="lfm2", question="tell me about simulated society")
    assert result.fell_back is True
    assert result.search_query == "simulated society"  # filler stripped even on fallback
    assert result.intent == "summarize_topic"


async def test_analyze_query_falls_back_on_unparseable_output():
    llm = _FakeLLM("I cannot help with that.")
    result = await analyze_query(llm, model="lfm2", question="what is retrieval augmented generation")
    assert result.fell_back is True
    assert result.search_query == "retrieval augmented generation"


async def test_url_in_question_forces_url_intent_even_if_model_misses_it():
    llm = _FakeLLM(
        '{"search_query": "mcp context", "keywords": "mcp context", '
        '"intent": "summarize_topic", "url": null}'
    )
    q = "Tell me more about this article: https://infoworld.com/article/123/mcp.html"
    result = await analyze_query(llm, model="lfm2", question=q)
    assert result.intent == "summarize_url"
    assert result.url == "https://infoworld.com/article/123/mcp.html"

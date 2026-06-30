"""Smoke tests for chunking strategy parsing — no DB or Ollama required."""

from superbrain.app.application.ingestion.chunking_agent import parse_strategy_response


def test_parse_strategy_response_valid():
    raw = '{"strategy": "semantic", "reason": "flowing prose"}'
    assert parse_strategy_response(raw) == "semantic"


def test_parse_strategy_response_with_markdown_fence():
    raw = '```json\n{"strategy": "recursive", "reason": "has headings"}\n```'
    assert parse_strategy_response(raw) == "recursive"


def test_parse_strategy_response_invalid_falls_back_to_fixed():
    raw = "I cannot determine the strategy."
    assert parse_strategy_response(raw) == "fixed"


def test_parse_strategy_response_unknown_strategy_falls_back():
    raw = '{"strategy": "unknown_value", "reason": "test"}'
    assert parse_strategy_response(raw) == "fixed"


def test_parse_strategy_response_fixed():
    raw = '{"strategy": "fixed", "reason": "long technical doc"}'
    assert parse_strategy_response(raw) == "fixed"


def test_parse_strategy_response_extra_whitespace():
    raw = '  \n  {"strategy": "semantic", "reason": "prose"}  \n  '
    assert parse_strategy_response(raw) == "semantic"

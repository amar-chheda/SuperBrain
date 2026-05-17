"""Smoke tests for classification response parsing — no DB or Ollama required."""

from uuid import UUID

from superbrain.app.application.topics.classifier import parse_classification_response

_TOPIC_ID_1 = UUID("aaaaaaaa-0000-0000-0000-000000000001")
_TOPIC_ID_2 = UUID("bbbbbbbb-0000-0000-0000-000000000002")

# index_map: integer index used in prompt → topic UUID
_INDEX_MAP = {1: _TOPIC_ID_1, 2: _TOPIC_ID_2}


def test_parse_classification_valid():
    raw = '[{"topic_id": 1, "confidence": "high", "reason": "directly relevant"}]'
    result = parse_classification_response(raw, _INDEX_MAP)
    assert len(result) == 1
    assert result[0].confidence == "high"
    assert result[0].topic_id == _TOPIC_ID_1


def test_parse_classification_rejects_unknown_index():
    raw = '[{"topic_id": 99, "confidence": "high", "reason": "test"}]'
    result = parse_classification_response(raw, _INDEX_MAP)
    assert len(result) == 0


def test_parse_classification_empty_array():
    raw = "[]"
    result = parse_classification_response(raw, {})
    assert result == []


def test_parse_classification_invalid_json_returns_empty():
    raw = "not json at all"
    result = parse_classification_response(raw, _INDEX_MAP)
    assert result == []


def test_parse_classification_invalid_confidence_coerced_to_low():
    raw = '[{"topic_id": 1, "confidence": "very_high", "reason": "test"}]'
    result = parse_classification_response(raw, _INDEX_MAP)
    assert len(result) == 1
    assert result[0].confidence == "low"


def test_parse_classification_with_markdown_fence():
    raw = '```json\n[{"topic_id": 1, "confidence": "medium", "reason": "ok"}]\n```'
    result = parse_classification_response(raw, _INDEX_MAP)
    assert len(result) == 1
    assert result[0].confidence == "medium"


def test_parse_classification_matches_object_wrapper():
    raw = '{"matches": [{"topic_id": 2, "confidence": "low", "reason": "tangential"}]}'
    result = parse_classification_response(raw, _INDEX_MAP)
    assert len(result) == 1
    assert result[0].topic_id == _TOPIC_ID_2

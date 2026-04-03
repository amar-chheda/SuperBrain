"""Integration tests for retrieval and grounded QA API."""

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from sqlalchemy import func, select

from superbrain.app.api.dependencies import get_chat_model_provider
from superbrain.app.application.ports import ChatModelProvider
from superbrain.app.application.qa.models import GeneratedAnswer
from superbrain.app.application.retrieval.models import EvidenceSet
from superbrain.app.config.settings import get_settings
from superbrain.app.infrastructure.db.base import Base
from superbrain.app.infrastructure.db.models import (
    ArticleChunkRecord,
    ArticleRecord,
    ModelCallLogRecord,
    QueryLogRecord,
)
from superbrain.app.infrastructure.db.session import (
    get_engine,
    get_session_factory,
    reset_db_caches,
)
from superbrain.app.main import create_app


class StubChatProvider(ChatModelProvider):
    """Deterministic chat provider for citation mapping tests."""

    def generate_answer(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer:
        """Return answer tied to first available evidence chunk."""

        if not evidence.chunks:
            return GeneratedAnswer(answer="insufficient evidence", supported=False)
        first_chunk_id = str(evidence.chunks[0].chunk.chunk_id)
        return GeneratedAnswer(
            answer=f"Answer for: {question}",
            supported=True,
            citation_chunk_ids=[first_chunk_id],
        )


def _setup_app(tmp_path: Path, monkeypatch: MonkeyPatch) -> TestClient:
    """Create app client with isolated SQLite database."""

    db_path = tmp_path / "test_qa.db"
    monkeypatch.setenv("SUPERBRAIN_DATABASE_URL", f"sqlite+pysqlite:///{db_path}")

    get_settings.cache_clear()
    reset_db_caches()

    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    return TestClient(create_app())


def _seed_chunks() -> None:
    """Seed article and chunk records for retrieval integration tests."""

    session_factory = get_session_factory()
    with session_factory() as session:
        article_id = uuid4()
        session.add(
            ArticleRecord(
                id=article_id,
                source_url="https://example.com/ai-systems",
                canonical_url="https://example.com/ai-systems",
                domain="example.com",
                title="AI Systems Overview",
                author=None,
                published_at=None,
                content="Reliable systems use observability and testing.",
                content_hash="hash-ai",
                extraction_quality_score=0.9,
                extraction_notes="seed",
            )
        )
        session.add_all(
            [
                ArticleChunkRecord(
                    id=uuid4(),
                    article_id=article_id,
                    chunk_index=0,
                    text="Observability includes logs, metrics, and tracing.",
                    token_count=6,
                    embedding=[0.9] * 16,
                    char_start=0,
                    char_end=52,
                ),
                ArticleChunkRecord(
                    id=uuid4(),
                    article_id=article_id,
                    chunk_index=1,
                    text="Testing prevents regressions in retrieval workflows.",
                    token_count=6,
                    embedding=[0.8] * 16,
                    char_start=53,
                    char_end=102,
                ),
            ]
        )
        session.commit()


def test_qa_endpoint_retrieves_seeded_chunks_and_returns_citations(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """QA endpoint should return grounded answer and citations from stored chunks."""

    client = _setup_app(tmp_path, monkeypatch)
    _seed_chunks()

    response = client.post(
        "/qa/ask",
        json={"question": "What helps retrieval workflows?", "top_k": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["supported"] is True
    assert len(body["citations"]) >= 1
    assert "article_url" in body["citations"][0]

    with get_engine().connect() as conn:
        query_log_count = conn.execute(
            select(func.count()).select_from(QueryLogRecord)
        ).scalar_one()
        model_call_log_count = conn.execute(
            select(func.count()).select_from(ModelCallLogRecord)
        ).scalar_one()

    assert query_log_count == 1
    assert model_call_log_count >= 2


def test_qa_endpoint_uses_mocked_generation_output_for_citations(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """QA endpoint should map model-selected citation chunk IDs to response citations."""

    client = _setup_app(tmp_path, monkeypatch)
    _seed_chunks()

    app = client.app
    app.dependency_overrides[get_chat_model_provider] = lambda: StubChatProvider()

    response = client.post(
        "/qa/ask",
        json={"question": "Tell me about observability", "top_k": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Answer for:")
    assert len(body["citations"]) == 1


def test_qa_endpoint_refuses_when_evidence_is_missing(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """QA endpoint should refuse unsupported claims when evidence is unavailable."""

    client = _setup_app(tmp_path, monkeypatch)

    response = client.post("/qa/ask", json={"question": "What is the GDP of Mars?", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["supported"] is False
    assert "enough evidence" in body["answer"].lower()
    assert body["citations"] == []

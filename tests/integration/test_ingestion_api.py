"""Integration tests for ingestion API workflow."""

from pathlib import Path

from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from sqlalchemy import func, select

from superbrain.app.api.dependencies import get_article_extractor
from superbrain.app.application.ports import ArticleExtractor, ExtractedArticle
from superbrain.app.config.settings import get_settings
from superbrain.app.infrastructure.db.base import Base
from superbrain.app.infrastructure.db.models import ArticleChunkRecord, ArticleRecord
from superbrain.app.infrastructure.db.session import get_engine, reset_db_caches
from superbrain.app.main import create_app


class StubExtractor(ArticleExtractor):
    """Deterministic extractor implementation for API integration tests."""

    def extract(self, url: str) -> ExtractedArticle:
        """Return fixed extraction output regardless of URL."""

        return ExtractedArticle(
            title="Sample Article",
            canonical_url=url,
            source_url=url,
            domain="example.com",
            author="Author",
            published_at=None,
            body_text=(
                "# Sample\n\n"
                "This is a test article body.\n\n"
                "It contains multiple paragraphs to create chunks."
            ),
            raw_html="<html><title>Sample</title></html>",
            extraction_quality_score=0.9,
            extraction_notes="stub",
        )


def _setup_app(tmp_path: Path, monkeypatch: MonkeyPatch) -> TestClient:
    """Create app client with isolated SQLite database."""

    db_path = tmp_path / "test_ingestion.db"
    monkeypatch.setenv("SUPERBRAIN_DATABASE_URL", f"sqlite+pysqlite:///{db_path}")

    get_settings.cache_clear()
    reset_db_caches()

    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    app = create_app()
    app.dependency_overrides[get_article_extractor] = lambda: StubExtractor()

    return TestClient(app)


def test_ingestion_api_persists_article_and_chunks(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Posting ingestion URL should persist article and chunk records."""

    client = _setup_app(tmp_path, monkeypatch)

    ingest_response = client.post("/ingestion/jobs", json={"url": "https://example.com/a?utm_source=x"})
    assert ingest_response.status_code == 200
    body = ingest_response.json()
    assert body["status"] == "succeeded"
    assert body["duplicate"] is False
    assert body["article_id"] is not None

    status_response = client.get(f"/ingestion/jobs/{body['job_id']}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "succeeded"

    with get_engine().connect() as conn:
        article_count = conn.execute(select(func.count()).select_from(ArticleRecord)).scalar_one()
        chunk_count = conn.execute(
            select(func.count()).select_from(ArticleChunkRecord)
        ).scalar_one()

    assert article_count == 1
    assert chunk_count >= 1


def test_ingestion_api_duplicate_submission_is_idempotent(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Submitting canonical duplicate URL should not create duplicate article rows."""

    client = _setup_app(tmp_path, monkeypatch)

    first = client.post("/ingestion/jobs", json={"url": "https://example.com/path/?utm_source=alpha"})
    second = client.post("/ingestion/jobs", json={"url": "https://example.com/path"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["duplicate"] is True

    with get_engine().connect() as conn:
        article_count = conn.execute(select(func.count()).select_from(ArticleRecord)).scalar_one()

    assert article_count == 1

"""Integration tests for digest generation and scheduler hooks."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from sqlalchemy import select

from superbrain.app.config.settings import get_settings
from superbrain.app.domain.models import TopicStatus
from superbrain.app.infrastructure.db.base import Base
from superbrain.app.infrastructure.db.models import (
    ArticleRecord,
    ArticleTopicMatchRecord,
    DigestRunRecord,
    ScheduledJobRecord,
    ScheduledJobRunRecord,
    TopicRecord,
    TopicVersionRecord,
)
from superbrain.app.infrastructure.db.session import (
    get_engine,
    get_session_factory,
    reset_db_caches,
)
from superbrain.app.main import create_app


def _setup_app(tmp_path: Path, monkeypatch: MonkeyPatch) -> TestClient:
    db_path = tmp_path / "test_digest.db"
    monkeypatch.setenv("SUPERBRAIN_DATABASE_URL", f"sqlite+pysqlite:///{db_path}")

    get_settings.cache_clear()
    reset_db_caches()

    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    return TestClient(create_app())


def _seed_digest_data() -> None:
    session_factory = get_session_factory()
    with session_factory() as session:
        yesterday = datetime.now(UTC) - timedelta(days=1)
        article_id = uuid4()
        topic_id = uuid4()
        topic_version_id = uuid4()

        session.add(
            ArticleRecord(
                id=article_id,
                source_url="https://example.com/digest-article",
                canonical_url="https://example.com/digest-article",
                domain="example.com",
                title="Digest Candidate",
                author=None,
                published_at=None,
                content="architecture reliability and testing",
                content_hash="digest-hash",
                extraction_quality_score=0.9,
                extraction_notes="seed",
                created_at=yesterday,
            )
        )

        session.add(
            TopicRecord(
                id=topic_id,
                name="Work",
                status=TopicStatus.ACTIVE.value,
                priority=10,
                current_version_id=topic_version_id,
                created_at=yesterday,
                updated_at=yesterday,
            )
        )

        session.add(
            TopicVersionRecord(
                id=topic_version_id,
                topic_id=topic_id,
                version_number=1,
                description="Work topics",
                positive_examples=["architecture"],
                negative_examples=["vacation"],
                created_at=yesterday,
            )
        )

        session.add(
            ArticleTopicMatchRecord(
                id=uuid4(),
                article_id=article_id,
                topic_id=topic_id,
                topic_version_id=topic_version_id,
                score=0.9,
                rationale="seed",
                disqualifiers=[],
                classified_at=yesterday,
            )
        )
        session.commit()


def test_digest_generation_over_seeded_data(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Digest trigger should generate grouped digest sections and persist runs."""

    client = _setup_app(tmp_path, monkeypatch)
    _seed_digest_data()

    response = client.post("/digests/trigger", json={})
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "succeeded"
    assert len(payload["items"]) >= 1
    assert payload["items"][0]["topic_name"] == "Work"

    with get_session_factory()() as session:
        runs = session.scalars(select(DigestRunRecord)).all()
        assert len(runs) == 1


def test_scheduler_manual_trigger_invokes_digest_job(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Scheduler trigger endpoint should execute registered digest jobs."""

    client = _setup_app(tmp_path, monkeypatch)
    _seed_digest_data()

    response = client.post("/digests/scheduler/trigger/daily_digest")
    assert response.status_code == 200
    assert response.json()["success"] is True

    with get_session_factory()() as session:
        runs = session.scalars(select(DigestRunRecord)).all()
        assert len(runs) >= 1
        scheduled = session.scalars(select(ScheduledJobRecord)).all()
        scheduled_runs = session.scalars(select(ScheduledJobRunRecord)).all()
        assert len(scheduled) >= 1
        assert len(scheduled_runs) >= 1


def test_scheduler_unknown_job_returns_404(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Unknown scheduler job names should map to a clean not-found API response."""

    client = _setup_app(tmp_path, monkeypatch)
    response = client.post("/digests/scheduler/trigger/not_registered")

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["code"] == "job_not_found"

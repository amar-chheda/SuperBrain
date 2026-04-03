"""Integration tests for topic APIs and classification workflows."""

from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from sqlalchemy import select

from superbrain.app.api.dependencies import get_topic_classifier
from superbrain.app.application.topics.classification import TopicClassifier
from superbrain.app.application.topics.models import (
    TopicClassificationDecision,
    TopicWithVersion,
)
from superbrain.app.config.settings import get_settings
from superbrain.app.infrastructure.db.base import Base
from superbrain.app.infrastructure.db.models import ArticleRecord, ArticleTopicMatchRecord
from superbrain.app.infrastructure.db.session import (
    get_engine,
    get_session_factory,
    reset_db_caches,
)
from superbrain.app.main import create_app


class StubTopicClassifier(TopicClassifier):
    """Deterministic classifier used for integration tests."""

    def classify(
        self,
        article_text: str,
        topics: list[TopicWithVersion],
    ) -> list[TopicClassificationDecision]:
        """Always match the first topic and reject the rest."""

        _ = article_text
        decisions: list[TopicClassificationDecision] = []
        for index, item in enumerate(topics):
            decisions.append(
                TopicClassificationDecision(
                    topic_id=item.topic.id,
                    topic_version_id=item.version.id,
                    matched=index == 0,
                    score=0.92 if index == 0 else 0.05,
                    rationale="stubbed classifier output",
                    disqualifiers=tuple() if index == 0 else ("not_relevant",),
                )
            )
        return decisions


def _setup_app(tmp_path: Path, monkeypatch: MonkeyPatch) -> TestClient:
    db_path = tmp_path / "test_topics.db"
    monkeypatch.setenv("SUPERBRAIN_DATABASE_URL", f"sqlite+pysqlite:///{db_path}")

    get_settings.cache_clear()
    reset_db_caches()

    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    app = create_app()
    app.dependency_overrides[get_topic_classifier] = lambda: StubTopicClassifier()
    return TestClient(app)


def _seed_article() -> str:
    session_factory = get_session_factory()
    article_id = uuid4()
    with session_factory() as session:
        session.add(
            ArticleRecord(
                id=article_id,
                source_url="https://example.com/research",
                canonical_url="https://example.com/research",
                domain="example.com",
                title="Research Notes",
                author=None,
                published_at=None,
                content="system design architecture testing reliability",
                content_hash="seed-hash",
                extraction_quality_score=0.8,
                extraction_notes="seed",
            )
        )
        session.commit()
    return str(article_id)


def test_classification_with_mocked_output_persists_matches(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Classifying an article should persist matched topic assignment."""

    client = _setup_app(tmp_path, monkeypatch)
    article_id = _seed_article()

    topic_response = client.post(
        "/topics",
        json={
            "name": "work",
            "description": "Work engineering material",
            "positive_examples": ["architecture", "testing"],
            "negative_examples": ["vacation"],
            "priority": 10,
        },
    )
    assert topic_response.status_code == 200

    classify_response = client.post(f"/topics/classify/articles/{article_id}")
    assert classify_response.status_code == 200
    matches = classify_response.json()
    assert len(matches) == 1
    assert matches[0]["score"] > 0.9

    with get_engine().connect() as conn:
        count = conn.execute(select(ArticleTopicMatchRecord)).all()
    assert len(count) == 1


def test_reclassification_after_topic_update_uses_new_version(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Reclassifying after topic update should store latest topic version ID."""

    client = _setup_app(tmp_path, monkeypatch)
    article_id = _seed_article()

    create_response = client.post(
        "/topics",
        json={
            "name": "engineering",
            "description": "Engineering docs",
            "positive_examples": ["design"],
            "negative_examples": ["recipe"],
            "priority": 20,
        },
    )
    assert create_response.status_code == 200
    created = create_response.json()
    topic_id = created["topic"]["id"]

    first_classify = client.post(f"/topics/classify/articles/{article_id}")
    assert first_classify.status_code == 200

    update_response = client.put(
        f"/topics/{topic_id}",
        json={
            "description": "Engineering and systems architecture docs",
            "positive_examples": ["architecture", "systems"],
            "negative_examples": ["travel"],
            "priority": 5,
        },
    )
    assert update_response.status_code == 200
    updated_version_id = update_response.json()["latest_version"]["id"]

    reclassify_response = client.post(
        "/topics/reclassify",
        json={"article_ids": [article_id], "limit": 10},
    )
    assert reclassify_response.status_code == 200
    assert reclassify_response.json()["processed_articles"] == 1

    with get_session_factory()() as session:
        stored_match = session.scalars(select(ArticleTopicMatchRecord)).first()
        assert stored_match is not None
        assert str(stored_match.topic_version_id) == updated_version_id

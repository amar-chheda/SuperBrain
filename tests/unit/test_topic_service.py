"""Unit tests for topic service behavior."""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from superbrain.app.application.topics.models import TopicCreateInput, TopicUpdateInput
from superbrain.app.application.topics.service import TopicService
from superbrain.app.domain.models import TopicDefinition, TopicStatus, TopicVersion


@dataclass
class InMemoryTopicRepository:
    """In-memory topic repository for unit testing topic service."""

    topics: dict[UUID, TopicDefinition]
    versions: dict[UUID, list[TopicVersion]]

    def create(self, topic: TopicDefinition, version: TopicVersion) -> TopicDefinition:
        self.topics[topic.id] = topic
        self.versions[topic.id] = [version]
        return topic

    def update(self, topic: TopicDefinition, version: TopicVersion) -> TopicDefinition:
        self.topics[topic.id] = topic
        self.versions[topic.id].append(version)
        return topic

    def set_inactive(self, topic_id: UUID) -> TopicDefinition:
        topic = self.topics[topic_id]
        updated = TopicDefinition(
            id=topic.id,
            name=topic.name,
            status=TopicStatus.INACTIVE,
            priority=topic.priority,
            current_version_id=topic.current_version_id,
            created_at=topic.created_at,
            updated_at=datetime.now(UTC),
        )
        self.topics[topic_id] = updated
        return updated

    def get(self, topic_id: UUID) -> TopicDefinition | None:
        return self.topics.get(topic_id)

    def list_all(self, active_only: bool = False) -> list[TopicDefinition]:
        values = list(self.topics.values())
        if active_only:
            return [value for value in values if value.status == TopicStatus.ACTIVE]
        return values

    def get_latest_version(self, topic_id: UUID) -> TopicVersion | None:
        entries = self.versions.get(topic_id, [])
        return entries[-1] if entries else None

    def list_active_with_latest_versions(self) -> list[tuple[TopicDefinition, TopicVersion]]:
        return [
            (topic, self.versions[topic.id][-1])
            for topic in self.topics.values()
            if topic.status == TopicStatus.ACTIVE
        ]


def test_topic_service_creates_and_updates_versions() -> None:
    """Updating a topic should create a new incremented version."""

    repository = InMemoryTopicRepository(topics={}, versions={})
    service = TopicService(repository)

    created = service.create_topic(
        TopicCreateInput(
            name="work",
            description="Engineering work topics",
            positive_examples=("design review",),
            negative_examples=("vacation",),
            priority=10,
        )
    )

    updated = service.update_topic(
        TopicUpdateInput(
            topic_id=created.id,
            description="Engineering work and architecture",
            positive_examples=("system design", "backend"),
            negative_examples=("personal finance",),
            priority=5,
        )
    )

    latest = service.get_latest_version(created.id)

    assert updated.priority == 5
    assert latest.version == 2
    assert updated.current_version_id == latest.id


def test_topic_service_deactivate() -> None:
    """Deactivating a topic should change status to inactive."""

    repository = InMemoryTopicRepository(topics={}, versions={})
    service = TopicService(repository)

    topic = service.create_topic(
        TopicCreateInput(
            name="personal",
            description="Personal life topics",
            positive_examples=("family",),
            negative_examples=("enterprise",),
            priority=50,
        )
    )

    deactivated = service.deactivate_topic(topic.id)
    assert deactivated.status == TopicStatus.INACTIVE

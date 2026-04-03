"""Topic lifecycle management service."""

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

from superbrain.app.application.topics.models import (
    TopicCreateInput,
    TopicUpdateInput,
    make_new_topic,
)
from superbrain.app.domain.models import TopicDefinition, TopicStatus, TopicVersion
from superbrain.app.domain.repositories import TopicRepository
from superbrain.app.errors import NotFoundError


class TopicService:
    """Manage topic CRUD and versioning behavior."""

    def __init__(self, topic_repository: TopicRepository) -> None:
        """Initialize service with repository boundary."""

        self._topic_repository = topic_repository

    def create_topic(self, input_data: TopicCreateInput) -> TopicDefinition:
        """Create topic and initial version."""

        topic, version = make_new_topic(input_data)
        return self._topic_repository.create(topic, version)

    def update_topic(self, input_data: TopicUpdateInput) -> TopicDefinition:
        """Create new topic version and update current metadata."""

        existing = self._topic_repository.get(input_data.topic_id)
        if existing is None:
            raise NotFoundError("topic not found")

        latest = self._topic_repository.get_latest_version(input_data.topic_id)
        if latest is None:
            raise NotFoundError("topic version not found")

        now = datetime.now(UTC)
        next_version = TopicVersion(
            id=uuid4(),
            topic_id=existing.id,
            version=latest.version + 1,
            description=input_data.description,
            positive_examples=input_data.positive_examples,
            negative_examples=input_data.negative_examples,
            created_at=now,
        )

        updated_topic = replace(
            existing,
            priority=input_data.priority,
            current_version_id=next_version.id,
            status=TopicStatus.ACTIVE,
            updated_at=now,
        )
        return self._topic_repository.update(updated_topic, next_version)

    def deactivate_topic(self, topic_id: UUID) -> TopicDefinition:
        """Deactivate topic and return updated topic state."""

        return self._topic_repository.set_inactive(topic_id)

    def list_topics(self, active_only: bool = False) -> list[TopicDefinition]:
        """List topics with optional active filter."""

        return self._topic_repository.list_all(active_only=active_only)

    def get_latest_version(self, topic_id: UUID) -> TopicVersion:
        """Fetch latest topic version for a topic ID."""

        version = self._topic_repository.get_latest_version(topic_id)
        if version is None:
            raise NotFoundError("topic version not found")
        return version

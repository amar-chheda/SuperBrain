"""Topic management and classification data models."""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from superbrain.app.domain.models import TopicDefinition, TopicStatus, TopicVersion


@dataclass(slots=True, frozen=True)
class TopicCreateInput:
    """Payload for creating a new topic."""

    name: str
    description: str
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    priority: int = 100


@dataclass(slots=True, frozen=True)
class TopicUpdateInput:
    """Payload for updating an existing topic."""

    topic_id: UUID
    description: str
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    priority: int


@dataclass(slots=True, frozen=True)
class TopicWithVersion:
    """Convenience model pairing topic metadata with latest version."""

    topic: TopicDefinition
    version: TopicVersion


@dataclass(slots=True, frozen=True)
class TopicClassificationDecision:
    """Per-topic classification decision for an article."""

    topic_id: UUID
    topic_version_id: UUID
    matched: bool
    score: float
    rationale: str
    disqualifiers: tuple[str, ...]


def make_new_topic(input_data: TopicCreateInput) -> tuple[TopicDefinition, TopicVersion]:
    """Construct initial topic aggregate with version 1."""

    now = datetime.now(UTC)
    topic_id = uuid4()
    version_id = uuid4()
    topic = TopicDefinition(
        id=topic_id,
        name=input_data.name,
        status=TopicStatus.ACTIVE,
        priority=input_data.priority,
        current_version_id=version_id,
        created_at=now,
        updated_at=now,
    )
    version = TopicVersion(
        id=version_id,
        topic_id=topic_id,
        version=1,
        description=input_data.description,
        positive_examples=input_data.positive_examples,
        negative_examples=input_data.negative_examples,
        created_at=now,
    )
    return topic, version

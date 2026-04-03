"""Unit tests for topic versioning logic."""

from superbrain.app.application.topics.models import TopicCreateInput, make_new_topic


def test_make_new_topic_creates_version_one() -> None:
    """Creating a topic aggregate should produce version number one."""

    topic, version = make_new_topic(
        TopicCreateInput(
            name="work",
            description="Work-related engineering and planning content",
            positive_examples=("system design",),
            negative_examples=("travel plans",),
            priority=10,
        )
    )

    assert topic.name == "work"
    assert version.version == 1
    assert version.topic_id == topic.id
    assert topic.current_version_id == version.id

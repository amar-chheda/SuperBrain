"""Groups articles by topic for the digest pipeline."""

from collections import defaultdict
from uuid import UUID

from superbrain.app.application.digest.types import ArticleWithTopics, TopicGroup
from superbrain.app.domain.entities import Topic

MIN_ARTICLES_PER_SECTION = 1


def group_by_topic(
    articles: list[ArticleWithTopics],
    topics: list[Topic],
) -> list[TopicGroup]:
    """Group articles by their matched topics, ordered by topic priority.

    Articles can appear in multiple groups if they matched multiple topics.
    Topics with no matching articles produce no group.
    """
    topic_map: dict[UUID, Topic] = {t.id: t for t in topics}
    groups: dict[UUID, list[ArticleWithTopics]] = defaultdict(list)

    for article in articles:
        for match in article.topic_matches:
            if match.confidence in ("high", "medium"):
                groups[match.topic_id].append(article)

    result = []
    for topic_id, group_articles in groups.items():
        topic = topic_map.get(topic_id)
        if not topic or len(group_articles) < MIN_ARTICLES_PER_SECTION:
            continue
        result.append(TopicGroup(topic=topic, articles=group_articles))

    result.sort(key=lambda g: (-g.topic.priority, -len(g.articles)))
    return result


def join_matches(
    articles: list,
    matches: list,
) -> list[ArticleWithTopics]:
    """Join Article entities with their ArticleTopicMatch lists."""
    from superbrain.app.application.digest.types import ArticleWithTopics

    match_map: dict[UUID, list] = defaultdict(list)
    for m in matches:
        match_map[m.article_id].append(m)

    return [
        ArticleWithTopics.from_article(a, match_map.get(a.id, []))
        for a in articles
    ]

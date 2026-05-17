"""Deduplicates sources within a topic group to avoid one domain dominating a section."""

import dataclasses
import re
from urllib.parse import urlparse

from superbrain.app.application.digest.types import ArticleWithTopics, TopicGroup

_CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def deduplicate_sources_within_group(group: TopicGroup) -> TopicGroup:
    """Keep only the highest-confidence article per domain within a topic group.

    This prevents a single news source from filling an entire digest section.
    It's a heuristic — occasionally discards genuinely distinct articles from
    the same domain, which is an acceptable tradeoff for digest quality.
    """
    seen_domains: dict[str, ArticleWithTopics] = {}

    for article in group.articles:
        domain = _extract_domain(article.url)
        existing = seen_domains.get(domain)

        if existing is None:
            seen_domains[domain] = article
        else:
            existing_conf = _best_confidence(existing, group.topic.id)
            new_conf = _best_confidence(article, group.topic.id)
            if new_conf > existing_conf:
                seen_domains[domain] = article

    return dataclasses.replace(group, articles=list(seen_domains.values()))


def _best_confidence(article: ArticleWithTopics, topic_id: object) -> int:
    return max(
        (_CONFIDENCE_ORDER.get(m.confidence, 0) for m in article.topic_matches if m.topic_id == topic_id),
        default=0,
    )


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return re.sub(r"^www\.", "", parsed.netloc.lower())

"""Article selection and deduplication for the digest pipeline."""

from datetime import date

from superbrain.app.application.digest.types import ArticleWithTopics


def select_articles_for_digest(
    articles: list[ArticleWithTopics],
    target_date: date,
) -> list[ArticleWithTopics]:
    """Return articles ingested on target_date with at least one high/medium match."""
    eligible = []
    for article in articles:
        if article.status != "succeeded":
            continue
        if article.ingested_at.date() != target_date:
            continue
        has_confident_match = any(
            m.confidence in ("high", "medium") for m in article.topic_matches
        )
        if not has_confident_match:
            continue
        eligible.append(article)

    return deduplicate_by_url(eligible)


def deduplicate_by_url(
    articles: list[ArticleWithTopics],
) -> list[ArticleWithTopics]:
    """Keep only the most recently ingested article per canonical URL."""
    seen: dict[str, ArticleWithTopics] = {}
    for article in sorted(articles, key=lambda a: a.ingested_at):
        seen[article.canonical_url] = article
    return list(seen.values())

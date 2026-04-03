"""Digest section generation logic."""

from collections import defaultdict

from superbrain.app.application.digest.models import DigestSectionDraft, DigestSourceArticle


class DigestGenerator:
    """Generate per-topic digest sections with citations and URLs."""

    def generate_sections(self, sources: list[DigestSourceArticle]) -> list[DigestSectionDraft]:
        """Create topic-grouped digest sections from source articles."""

        grouped: dict[tuple[str, str], list[DigestSourceArticle]] = defaultdict(list)
        for source in sources:
            key = (
                str(source.topic_id) if source.topic_id is not None else "none",
                source.topic_name,
            )
            grouped[key].append(source)

        sections: list[DigestSectionDraft] = []
        for group_sources in grouped.values():
            representative = group_sources[0]
            titles = [entry.article.title for entry in group_sources[:3]]
            summary = f"{len(group_sources)} article(s): " + "; ".join(titles)

            sections.append(
                DigestSectionDraft(
                    topic_id=representative.topic_id,
                    topic_name=representative.topic_name,
                    summary=summary,
                    source_urls=tuple(entry.article.source_url for entry in group_sources),
                    citation_article_ids=tuple(entry.article.id for entry in group_sources),
                )
            )

        sections.sort(key=lambda section: section.topic_name)
        return sections

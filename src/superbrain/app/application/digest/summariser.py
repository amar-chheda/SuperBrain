"""LLM-based summarisation for digest topic groups."""

from superbrain.app.application.digest.types import ArticleWithTopics, TopicGroup
from superbrain.app.application.ports import LLMPort

_MAX_EXCERPT_CHARS = 400
_MAX_ARTICLES = 5

DIGEST_SUMMARY_PROMPT = """You are a news digest writer. Your job is to write a brief, informative summary of the following articles about {topic_name}.

TOPIC: {topic_name}
TOPIC DESCRIPTION: {topic_description}

ARTICLES:
{articles_block}

RULES:
- Write a summary of 3 to 6 sentences covering the key developments across these articles
- Synthesise the articles — do not summarise each one individually
- Focus on what is new, significant, or actionable
- Write in a neutral, factual tone
- Do not include phrases like "According to the articles" or "The articles discuss"
- Do not use bullet points — write in prose
- Do not hallucinate any information not present in the articles above

Write the summary now:"""


def format_articles_block(articles: list[ArticleWithTopics]) -> str:
    lines = []
    for i, article in enumerate(articles[:_MAX_ARTICLES], start=1):
        lines.append(f"Article {i}: {article.title or article.url}")
        excerpt = article.raw_text[:_MAX_EXCERPT_CHARS].replace("\n", " ").strip()
        lines.append(excerpt)
        lines.append("")
    return "\n".join(lines)


async def summarise_topic_group(
    llm: LLMPort,
    model: str,
    group: TopicGroup,
) -> str:
    """Generate a prose summary for one topic group. Returns empty string if no articles."""
    if not group.articles:
        return ""

    prompt = DIGEST_SUMMARY_PROMPT.format(
        topic_name=group.topic.name,
        topic_description=group.topic.description,
        articles_block=format_articles_block(group.articles),
    )

    summary = await llm.complete(
        prompt,
        model=model,
        prompt_template="digest_summary_v1",
    )
    return summary.strip()

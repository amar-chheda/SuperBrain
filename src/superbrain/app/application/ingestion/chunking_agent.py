"""Chunking strategy agent.

Uses a local LLM to inspect article metadata and decide which chunking
strategy to apply. This is the first agentic decision in the system.

The prompt is written defensively for 8B models: explicit decision rules,
exact output format, and a safe fallback for malformed responses.
"""

import json
import re
from typing import Literal

import structlog

from superbrain.app.application.ports import LLMPort

log = structlog.get_logger(__name__)

CHUNKING_STRATEGY_PROMPT = """You are a text chunking strategy selector. Your job is to read the article metadata below and return a JSON object specifying which chunking strategy to use.

ARTICLE METADATA:
- Title: {title}
- Word count: {word_count}
- First 500 characters: {preview}
- Detected structure: {structure_hints}

CHUNKING STRATEGIES AVAILABLE:
1. "semantic" — splits on sentence boundaries and groups by meaning
   USE WHEN: the text is flowing prose, news articles, essays, blog posts
   DO NOT USE WHEN: the text has lists, tables, code, or clear section headers

2. "recursive" — splits on paragraph breaks, then sentences, then words
   USE WHEN: the text has clear paragraph structure but mixed content types
   USE WHEN: articles with some lists or subheadings mixed with prose

3. "fixed" — splits into fixed token windows with overlap
   USE WHEN: the text is very long (over 5000 words) and structure is unclear
   USE WHEN: the text contains code, technical documentation, or tabular data
   USE WHEN: you are uncertain which other strategy to use

DECISION RULES (apply in order, stop at first match):
- If word_count > 5000 → use "fixed"
- If structure_hints contains "code_block" or "table" → use "fixed"
- If structure_hints contains "heading" and word_count < 5000 → use "recursive"
- Otherwise → use "semantic"

YOU MUST respond with ONLY a valid JSON object. No explanation. No preamble. No markdown.
The JSON must have exactly this structure:
{{"strategy": "semantic" | "recursive" | "fixed", "reason": "one sentence"}}

Example valid response:
{{"strategy": "semantic", "reason": "flowing prose news article under 5000 words"}}"""


def detect_structure(text: str) -> list[str]:
    """Heuristic scan for document structure signals.

    No LLM needed — cheap regex checks that inform the strategy prompt.

    Args:
        text: The article text to inspect.

    Returns:
        List of detected structure types (e.g. ['heading', 'list']).
    """
    hints: list[str] = []
    if re.search(r"^#{1,3}\s", text, re.MULTILINE):
        hints.append("heading")
    if "```" in text or re.search(r"^    \S", text, re.MULTILINE):
        hints.append("code_block")
    if re.search(r"^\s*[\|\+][-+|]+", text, re.MULTILINE):
        hints.append("table")
    if re.search(r"^\s*[-*]\s", text, re.MULTILINE):
        hints.append("list")
    return hints


def parse_strategy_response(raw: str) -> Literal["semantic", "recursive", "fixed"]:
    """Parse the LLM's JSON response defensively.

    Handles common local model misbehaviours: markdown fences, extra whitespace,
    trailing commas. Falls back to 'fixed' on any parse failure — conservative
    but always produces output.

    Args:
        raw: The raw string returned by the LLM.

    Returns:
        One of 'semantic', 'recursive', or 'fixed'.
    """
    valid = {"semantic", "recursive", "fixed"}

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    try:
        parsed = json.loads(cleaned)
        strategy = parsed.get("strategy", "")
        if strategy in valid:
            return strategy  # type: ignore[return-value]
        log.warning("chunking_agent.invalid_strategy", strategy=strategy)
    except (json.JSONDecodeError, AttributeError) as exc:
        log.warning("chunking_agent.parse_failed", error=str(exc), raw=raw[:200])

    return "fixed"


async def decide_chunking_strategy(
    llm: LLMPort,
    model: str,
    article_text: str,
    title: str | None,
    related_entity_id: "object | None" = None,
) -> Literal["semantic", "recursive", "fixed"]:
    """Ask the LLM to choose a chunking strategy for this article.

    Args:
        llm: The LLM port implementation to call.
        model: Ollama model tag to use for the decision.
        article_text: Full article text.
        title: Article title, or None if unknown.
        related_entity_id: Article UUID for audit log correlation.

    Returns:
        The chosen strategy: 'semantic', 'recursive', or 'fixed'.
    """
    from uuid import UUID

    word_count = len(article_text.split())
    preview = article_text[:500].replace("{", "{{").replace("}", "}}")
    structure_hints = detect_structure(article_text)

    prompt = CHUNKING_STRATEGY_PROMPT.format(
        title=title or "Unknown",
        word_count=word_count,
        preview=preview,
        structure_hints=", ".join(structure_hints) if structure_hints else "none detected",
    )

    entity_id = related_entity_id if isinstance(related_entity_id, UUID) else None

    raw = await llm.complete(
        prompt,
        model=model,
        json_mode=True,
        prompt_template="chunking_strategy_v1",
        related_entity_id=entity_id,
    )

    strategy = parse_strategy_response(raw)
    log.info("chunking_agent.decided", strategy=strategy, word_count=word_count,
             structure=structure_hints)
    return strategy

"""Article topic classifier using a local LLM.

Uses Phi-3 Mini (settings.ollama_classification_model) to classify articles
against batches of topics. Topics are identified by sequential integers in the
prompt (1, 2, 3...) rather than UUIDs because local models hallucinate long IDs.
The integer is mapped back to the real UUID after parsing.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import structlog

from superbrain.app.application.ports import LLMPort
from superbrain.app.domain.entities import Article, Topic

log = structlog.get_logger(__name__)

CLASSIFICATION_PROMPT = """### ROLE
You are a strict content classification engine. You read an article and decide which topics from a given list it belongs to. You output only structured JSON — never prose, never explanation outside the JSON.

### ARTICLE TO CLASSIFY
Title: {title}

Excerpt (first 800 characters of body text):
{excerpt}

### TOPICS LIST
Each topic has a numeric ID, a name, a description, and optional examples.
{topics_block}

### YOUR JOB
Go through each topic one by one. For each topic ask yourself: "Is the main subject of this article directly and substantially about this topic?"

Use these confidence levels:
- "high"   — the article is clearly and primarily focused on this topic; the topic appears in the title or is the central theme
- "medium" — the article covers this topic in significant depth, even if it is not the sole focus
- "low"    — the article touches this topic in a meaningful way but it is secondary to the main subject

DO NOT match a topic if:
- The topic is only mentioned in passing (one sentence or a brief reference)
- The article is only tangentially related
- You are not confident the article would be useful to someone interested in that topic

An article may match zero topics, one topic, or several topics. All are valid outcomes.

### OUTPUT FORMAT
You must output ONLY a JSON object with a single key "matches". Do not write anything before or after the JSON. Do not use markdown code fences. Do not add any commentary.

The "matches" value is an array of objects. Each object must have exactly these three fields:
  "topic_id"   — the integer ID of the matching topic (just the number, e.g. 1 or 2)
  "confidence" — one of the three strings: "high", "medium", or "low"
  "reason"     — a single sentence (10–25 words) explaining specifically why the article matches this topic

### EXAMPLES

Example 1 — article matches one topic with high confidence:
Topics available: 1=Python Programming, 2=Cloud Infrastructure, 3=Cybersecurity
Article: "Python 3.12 released with major performance improvements and new syntax features"
Correct output:
{{"matches": [{{"topic_id": 1, "confidence": "high", "reason": "The article is entirely focused on a new Python release and its language improvements."}}]}}

Example 2 — article matches two topics:
Topics available: 1=Python Programming, 2=Cloud Infrastructure, 3=Cybersecurity
Article: "How to deploy a Python Flask app securely on AWS with IAM roles and VPC isolation"
Correct output:
{{"matches": [{{"topic_id": 1, "confidence": "medium", "reason": "The article uses Python Flask as the application being deployed, making Python central to the guide."}}, {{"topic_id": 2, "confidence": "high", "reason": "The article is primarily about deploying and securing a cloud application on AWS infrastructure."}}]}}

Example 3 — article matches no topics:
Topics available: 1=Python Programming, 2=Cloud Infrastructure, 3=Cybersecurity
Article: "The history of the Roman Empire and its influence on modern legal systems"
Correct output:
{{"matches": []}}

Example 4 — do not match on passing mentions:
Topics available: 1=Electric Vehicles, 2=Battery Technology
Article: "The stock market had a volatile week; Tesla and Rivian shares dropped 8% while oil stocks surged. Analysts cite interest rate fears."
The article mentions EV companies but is really about stock market movements. Correct output:
{{"matches": []}}

Example 5 — low confidence is still a valid match:
Topics available: 1=Machine Learning, 2=Data Engineering
Article: "Building a real-time data pipeline with Apache Kafka and dbt for analytics teams"
The article is primarily about data engineering, but mentions that the pipeline feeds an ML feature store.
Correct output:
{{"matches": [{{"topic_id": 1, "confidence": "low", "reason": "The pipeline is described as feeding an ML feature store, giving machine learning a secondary but real role."}}, {{"topic_id": 2, "confidence": "high", "reason": "The article is entirely about building a real-time data pipeline using Kafka and dbt."}}]}}

### NOW CLASSIFY THE ARTICLE ABOVE
Output only the JSON object. Start your response with {{ and end with }}."""  # noqa: E501


@dataclass
class TopicMatch:
    """Ephemeral value object returned by the classifier before persistence.

    Attributes:
        topic_id: UUID of the matched topic.
        confidence: Classifier's confidence level.
        reason: One-sentence explanation of why the article matches.
    """

    topic_id: UUID
    confidence: Literal["high", "medium", "low"]
    reason: str


def format_topics_block(topics: list[Topic], index_offset: int = 0) -> str:
    """Format topics for the classification prompt using sequential integer IDs.

    Uses integers (1, 2, 3...) instead of UUIDs because local models hallucinate
    long UUIDs. The caller is responsible for mapping integers back to UUIDs via
    build_index_map().

    Args:
        topics: Topics to format.
        index_offset: Starting offset so batches produce globally unique IDs.

    Returns:
        Formatted string block for prompt injection.
    """
    lines: list[str] = []
    for i, t in enumerate(topics, start=index_offset + 1):
        lines.append(f"ID: {i}")
        lines.append(f"Name: {t.name}")
        lines.append(f"Description: {t.description}")
        if t.examples:
            lines.append(f"Examples: {', '.join(t.examples[:3])}")
        lines.append("")
    return "\n".join(lines)


def build_index_map(topics: list[Topic], index_offset: int = 0) -> dict[int, UUID]:
    """Build a mapping from prompt integer IDs to topic UUIDs.

    Args:
        topics: Topics in the same order passed to format_topics_block.
        index_offset: Same offset used in format_topics_block.

    Returns:
        Dict mapping integer prompt ID → topic UUID.
    """
    return {i: t.id for i, t in enumerate(topics, start=index_offset + 1)}


def parse_classification_response(
    raw: str, index_map: dict[int, UUID]
) -> list[TopicMatch]:
    """Parse and validate the classifier's JSON output defensively.

    Local models sometimes emit markdown fences or invalid JSON. This function
    handles those cases. Returns an empty list on any parse failure — never
    crashes ingestion.

    Args:
        raw: Raw string returned by the LLM.
        index_map: Mapping from prompt integer IDs to topic UUIDs.

    Returns:
        List of validated TopicMatch objects. May be empty.
    """
    try:
        cleaned = raw.strip()
        # Strip markdown code fences if present
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

        parsed = json.loads(cleaned)

        # Accept {"matches": [...]} (preferred) or a bare list (fallback)
        if isinstance(parsed, dict):
            parsed = parsed.get("matches", [])
        if not isinstance(parsed, list):
            log.warning("classification.invalid_response", reason="not a list", raw=raw[:200])
            return []

        matches: list[TopicMatch] = []
        for item in parsed:
            raw_id = item.get("topic_id")
            confidence = item.get("confidence", "")
            reason = item.get("reason", "")

            try:
                numeric_id = int(raw_id)
            except (TypeError, ValueError):
                log.warning("classification.unknown_topic_id", topic_id=raw_id)
                continue

            topic_uuid = index_map.get(numeric_id)
            if topic_uuid is None:
                log.warning("classification.unknown_topic_id", topic_id=raw_id)
                continue

            if confidence not in ("high", "medium", "low"):
                log.warning("classification.invalid_confidence", confidence=confidence)
                confidence = "low"

            matches.append(TopicMatch(
                topic_id=topic_uuid,
                confidence=confidence,  # type: ignore[arg-type]
                reason=reason or "No reason provided",
            ))

        return matches

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        log.warning("classification.parse_failed", error=str(exc), raw=raw[:200])
        return []


async def classify_article(
    llm: LLMPort,
    model: str,
    article: Article,
    topics: list[Topic],
    batch_size: int = 5,
) -> list[TopicMatch]:
    """Classify an article against all active topics.

    Sends topics in batches to stay within the local model's context window.
    Uses sequential integer IDs in the prompt to avoid UUID hallucination.
    Results from all batches are combined and returned.

    Args:
        llm: The LLM port implementation.
        model: Ollama model tag to use.
        article: The article to classify.
        topics: All active topics to classify against.
        batch_size: Number of topics per LLM call. Default 5.

    Returns:
        All topic matches across all batches. May be empty.
    """
    if not topics:
        return []

    all_matches: list[TopicMatch] = []

    for batch_start in range(0, len(topics), batch_size):
        batch = topics[batch_start: batch_start + batch_size]
        index_map = build_index_map(batch, index_offset=batch_start)

        prompt = CLASSIFICATION_PROMPT.format(
            title=article.title or "Unknown",
            excerpt=article.raw_text[:800],
            topics_block=format_topics_block(batch, index_offset=batch_start),
        )

        topic_names = [t.name for t in batch]
        log.info(
            "classification.batch_sending",
            model=model,
            batch_start=batch_start,
            topic_ids=[i for i in index_map],
            topic_names=topic_names,
        )

        raw = await llm.complete(
            prompt,
            model=model,
            json_mode=True,
            prompt_template="article_classification_v1",
            related_entity_id=article.id,
        )

        log.info("classification.model_raw_response", raw_response=raw)

        batch_matches = parse_classification_response(raw, index_map)
        all_matches.extend(batch_matches)
        log.info(
            "classification.batch_done",
            batch_start=batch_start,
            batch_size=len(batch),
            matches=len(batch_matches),
            results=[
                {"topic_id": str(m.topic_id), "confidence": m.confidence, "reason": m.reason}
                for m in batch_matches
            ],
        )

    return all_matches

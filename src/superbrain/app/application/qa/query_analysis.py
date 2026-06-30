"""Query analysis: decompose a user question into routed parts before retrieval.

A small thinking model (lfm2.5-thinking) splits the raw ask into parts that each
go to a *different* stage:

    search_query        -> vector probe + reranker query   (clean retrieval topic)
    keywords            -> BM25 lexical search              (OR-semantics terms)
    hypothetical_passage -> vector HyDE probe               (pool-widening only)
    answer_directives   -> the generation prompt ONLY       (how to shape the answer)
    intent / url        -> routing (topic search vs direct article lookup)

The user's literal text is never embedded or reranked directly — only the
extracted search_query is. The model output is used ONLY to find/shape, never as
fact, so grounding and citations are preserved. Every failure mode degrades:
LLM down / bad JSON / hallucinated topic -> a deterministic strip of the raw
question, so the pipeline never breaks and never embeds the filler.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

import structlog

from superbrain.app.application.ports import LLMPort

log = structlog.get_logger(__name__)

Intent = Literal["summarize_topic", "summarize_url"]

_URL_RE = re.compile(r"https?://[^\s)>\]}\"']+")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(raw: str) -> str:
    cleaned = _THINK_RE.sub("", raw)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned

# Leading carrier/filler phrases ("tell me more about", "explain", ...).
_FILLER_RE = re.compile(
    r"^\s*(please\s+)?(can|could|would|will)?\s*(you\s+)?(please\s+)?"
    r"(tell\s+me(\s+more)?(\s+about)?|explain(\s+to\s+me)?|describe|summari[sz]e|"
    r"give\s+me(\s+a)?(\s+summary(\s+of)?)?|walk\s+me\s+through|"
    r"what(\s+is|\s+are|'s)?(\s+the)?|who(\s+is|\s+are)?|"
    r"i(\s+want|'d\s+like)\s+to\s+know(\s+about)?|help\s+me\s+understand)\b[:,]?\s*",
    re.IGNORECASE,
)

# Answer-shaping instructions — extracted to answer_directives, removed from topic.
_DIRECTIVE_RE = re.compile(
    r"("
    r"\bbe\s+(more\s+)?(detailed|concise|brief|thorough|specific|verbose|short|clear)\b"
    r"|\bkeep\s+(it|your\s+answer|the\s+response|things?)\s+\w+(\s+and\s+\w+)?"
    r"|\bmake\s+(it|your\s+answer|the\s+response)\s+\w+"
    r"|\b(use|in|as|with|format(ted)?\s+(it\s+)?(as|in|with))\s+"
    r"(bullet\s*points?|bullets|a\s+list|lists?|a\s+table|tables?|paragraphs?|prose|markdown)\b"
    r"|\b(well[\s-]?)?structured?\b"
    r"|\bstep[\s-]by[\s-]step\b"
    r"|\bin\s+\d+\s+(sentences?|paragraphs?|words?|points?|bullets?)\b"
    r"|\b(include|with|add|give)\s+(examples?|citations?|sources?|references?)\b"
    r")",
    re.IGNORECASE,
)


@dataclass
class QueryAnalysis:
    """Structured, routed decomposition of a user question."""

    search_query: str          # clean retrieval topic -> vector probe + reranker query
    keywords: str              # core terms -> BM25
    hypothetical_passage: str  # HyDE probe -> vector recall only
    answer_directives: str     # output-shaping instructions -> generation prompt only
    intent: Intent
    url: str | None
    fell_back: bool = False    # True if model analysis failed and we used the deterministic strip


_PROMPT = """You preprocess a user's question before a document search. Extract its parts and respond with ONE JSON object and nothing else.

USER QUESTION:
{question}

Rules:
- "search_query": the core topic to search for, COPIED from the question with only filler ("tell me about", "explain") and answer-formatting instructions removed. Do NOT add words that are not in the question.
- "keywords": 2-5 of the most important search terms from the topic.
- "answer_directives": instructions about HOW to shape the answer (e.g. "be detailed", "use bullet points", "keep it structured"), copied from the question. Empty string if none.
- "hypothetical_passage": a 1-2 sentence passage a relevant article might contain about the topic. Stay on-topic; invent no specific facts, names, or numbers.
- "intent": "summarize_url" if it targets a specific article or link, otherwise "summarize_topic".
- "url": the URL in the question if there is one, otherwise null.

Example:
question: "tell me more about the simulated society experiment — be detailed and use bullet points"
{{"search_query": "simulated society experiment", "keywords": "simulated society experiment", "answer_directives": "be detailed; use bullet points", "hypothetical_passage": "A simulated society experiment runs many AI agents together to study emergent social behavior.", "intent": "summarize_topic", "url": null}}

JSON:"""


def detect_url(question: str) -> str | None:
    """Return the first URL in the question (punctuation-trimmed), or None."""
    match = _URL_RE.search(question)
    if not match:
        return None
    return match.group(0).rstrip(".,;:!?")


def _deterministic_split(question: str) -> tuple[str, str]:
    """Heuristically split a question into (clean_topic, answer_directives).

    Used as the backstop whenever the LLM is unavailable, returns bad JSON, or
    hallucinates a topic. Conservative: when in doubt it keeps the original text.
    """
    directives = [m.group(0).strip() for m in _DIRECTIVE_RE.finditer(question)]
    topic = _DIRECTIVE_RE.sub(" ", question)
    topic = _FILLER_RE.sub("", topic, count=1)
    topic = re.sub(r"\s+", " ", topic).strip(" .,:;-—?!\t")
    topic = re.sub(r"\s+\b(and|,)\s*$", "", topic).strip(" .,:;-—?!")
    if not topic:
        topic = question.strip()
    directives_str = "; ".join(dict.fromkeys(d.lower() for d in directives))
    return topic, directives_str


def _is_subset_of_question(text: str, question: str) -> bool:
    """True if `text` is mostly drawn from `question` (i.e. not hallucinated)."""
    q = set(re.findall(r"\w+", question.lower()))
    t = set(re.findall(r"\w+", text.lower()))
    if not t:
        return False
    novel = [w for w in t if w not in q]
    return len(novel) / len(t) <= 0.34


def raw_analysis(question: str) -> QueryAnalysis:
    """Deterministic decomposition with no LLM (used when analysis is disabled)."""
    topic, directives = _deterministic_split(question)
    url = detect_url(question)
    return QueryAnalysis(
        search_query=topic,
        keywords=topic,
        hypothetical_passage=topic,
        answer_directives=directives,
        intent="summarize_url" if url else "summarize_topic",
        url=url,
        fell_back=False,
    )


def _fallback(question: str) -> QueryAnalysis:
    """Deterministic decomposition, flagged as a fallback (LLM failed)."""
    analysis = raw_analysis(question)
    analysis.fell_back = True
    return analysis


def _extract_json(raw: str) -> dict | None:
    """Strip <think> blocks and parse the first balanced JSON object found."""
    cleaned = _strip_think(raw)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


async def analyze_query(llm: LLMPort, *, model: str, question: str) -> QueryAnalysis:
    """Decompose the question with the thinking model; degrade to a deterministic split."""
    try:
        raw = await llm.complete(
            _PROMPT.format(question=question),
            model=model,
            prompt_template="query_analysis_v1",
        )
    except Exception as exc:  # LLMError or anything else — never break QA
        log.warning("query_analysis.llm_failed", error=str(exc), question=question[:100])
        return _fallback(question)

    data = _extract_json(raw)
    if data is None:
        log.warning("query_analysis.parse_failed", raw=raw[:200], question=question[:100])
        return _fallback(question)

    det_topic, det_directives = _deterministic_split(question)

    search_query = str(data.get("search_query") or "").strip()
    if not search_query or not _is_subset_of_question(search_query, question):
        # LLM drifted (added words not in the question) or returned nothing —
        # trust the deterministic strip rather than embed a hallucinated topic.
        log.info("query_analysis.search_query_rejected", llm=search_query[:80], used=det_topic[:80])
        search_query = det_topic

    kw = data.get("keywords")
    if isinstance(kw, list):  # models sometimes return a JSON array instead of a string
        kw = " ".join(str(x) for x in kw)
    keywords = str(kw or "").strip() or search_query
    directives = str(data.get("answer_directives") or "").strip() or det_directives
    passage = str(data.get("hypothetical_passage") or "").strip() or search_query

    intent = data.get("intent")
    if intent not in ("summarize_topic", "summarize_url"):
        intent = "summarize_topic"

    # Deterministic URL detection wins: a 1.2B model often misses or mangles URLs.
    url = detect_url(question) or (str(data["url"]).strip() if data.get("url") else None)
    if url:
        intent = "summarize_url"

    return QueryAnalysis(
        search_query=search_query,
        keywords=keywords,
        hypothetical_passage=passage,
        answer_directives=directives,
        intent=intent,  # type: ignore[arg-type]
        url=url,
        fell_back=False,
    )

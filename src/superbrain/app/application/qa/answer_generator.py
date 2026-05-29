"""Grounded answer generation using retrieved evidence chunks."""

import re
from uuid import UUID

import structlog

from superbrain.app.application.ports import LLMPort
from superbrain.app.application.qa.evidence_builder import Evidence

log = structlog.get_logger(__name__)

GROUNDED_QA_PROMPT = """You are a question-answering assistant. Answer using ONLY the numbered evidence below.

QUESTION:
{question}

EVIDENCE:
{evidence_block}

RULES:
- Cite sources inline using their number, like [1] or [2], immediately after the sentence that uses that source.
- Use ONLY information present in the evidence above. No outside knowledge.
- If the evidence does not contain enough information, say exactly: "I cannot answer this question based on the available evidence."
- Keep your answer concise — 2 to 5 sentences unless more detail is needed.
- End your response with a SOURCES line listing only the numbers you cited, like:
  SOURCES: 1, 2

Begin your answer now:"""


def format_evidence_block(evidence: list[Evidence]) -> str:
    lines = []
    for i, e in enumerate(evidence, start=1):
        lines.append(f"[{i}] {e.article_title or e.article_url}")
        lines.append(e.content)
        lines.append("")
    return "\n".join(lines)


async def generate_answer(
    llm: LLMPort,
    model: str,
    question: str,
    evidence: list[Evidence],
) -> tuple[str, list[tuple[int, UUID]], str]:
    """Generate a grounded answer and return (answer_text, cited_pairs, prompt_sent).

    cited_pairs is a list of (citation_number, chunk_id) in the order cited.
    """
    prompt = GROUNDED_QA_PROMPT.format(
        question=question,
        evidence_block=format_evidence_block(evidence),
    )
    raw = await llm.complete(prompt, model=model, prompt_template="grounded_qa_v1")
    answer_text, cited_pairs = parse_answer_response(raw, evidence)
    return answer_text, cited_pairs, prompt


def parse_answer_response(
    raw: str, evidence: list[Evidence]
) -> tuple[str, list[tuple[int, UUID]]]:
    """Split model output into answer text and (citation_number, chunk_id) pairs.

    Numbers in SOURCES map to 1-based indices into the evidence list.
    Out-of-range numbers are logged and dropped.
    """
    parts = re.split(r"\nSOURCES:\s*", raw, maxsplit=1)
    answer_text = parts[0].strip()

    if len(parts) < 2:
        log.warning("qa.missing_sources_line", raw=raw[:200])
        return answer_text, []

    cited_pairs: list[tuple[int, UUID]] = []
    seen: set[int] = set()
    for token in re.split(r"[\s,]+", parts[1].strip()):
        token = token.strip("[].,")
        if not token.isdigit():
            continue
        n = int(token)
        if n < 1 or n > len(evidence):
            log.warning("qa.citation_out_of_range", number=n, evidence_count=len(evidence))
            continue
        if n not in seen:
            seen.add(n)
            cited_pairs.append((n, evidence[n - 1].chunk_id))

    return answer_text, cited_pairs

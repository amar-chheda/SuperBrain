"""Grounded answer generation using retrieved evidence chunks."""

from __future__ import annotations

import asyncio
import re
from uuid import UUID

import dspy
import structlog

from superbrain.app.application.ports import LLMPort
from superbrain.app.application.qa.evidence_builder import Evidence
from superbrain.app.infrastructure.llm.dspy_lm import LLMPortLM

log = structlog.get_logger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")


class _GroundedQA(dspy.Signature):
    """Answer using ONLY the numbered evidence below.

    Cite sources inline using their number, like [1] or [2], immediately after the
    sentence that uses that source. Use ONLY information present in the evidence —
    no outside knowledge. If the evidence does not contain enough information, say
    exactly: "I cannot answer this question based on the available evidence."
    """

    question: str = dspy.InputField()
    evidence: str = dspy.InputField(desc="numbered evidence passages, [1], [2], ...")
    format_directives: str = dspy.InputField(
        desc="how to shape the answer's length/format — never changes what counts as evidence"
    )
    answer: str = dspy.OutputField(
        desc="the grounded answer, with inline [n] citations right after each sentence "
        "that uses source n"
    )


_answer_question = dspy.Predict(_GroundedQA)


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
    answer_directives: str = "",
) -> tuple[str, list[tuple[int, UUID]], str]:
    """Generate a grounded answer and return (answer_text, cited_pairs, prompt_sent).

    cited_pairs is a list of (citation_number, chunk_id) in the order cited.
    answer_directives are the user's output-shaping instructions (e.g. "be detailed",
    "use bullet points") extracted upstream — they shape HOW the answer reads, never
    what counts as evidence.
    """
    directives = (answer_directives or "").strip()
    format_directives = directives or (
        "Keep your answer concise — 2 to 5 sentences unless more detail is needed."
    )

    stage_lm = LLMPortLM(llm, model=model, prompt_template="grounded_qa_v1")
    with dspy.context(lm=stage_lm, adapter=dspy.JSONAdapter()):
        result = await asyncio.to_thread(
            _answer_question,
            question=question,
            evidence=format_evidence_block(evidence),
            format_directives=format_directives,
        )
    answer_text, cited_pairs = parse_answer_response(result.answer, evidence)
    return answer_text, cited_pairs, stage_lm.last_prompt or ""


def parse_answer_response(
    answer: str, evidence: list[Evidence]
) -> tuple[str, list[tuple[int, UUID]]]:
    """Extract (citation_number, chunk_id) pairs from inline [n] citations in the answer.

    Numbers map to 1-based indices into the evidence list, in first-appearance order.
    Out-of-range numbers are logged and dropped.
    """
    cited_pairs: list[tuple[int, UUID]] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        n = int(match.group(1))
        if n < 1 or n > len(evidence):
            log.warning("qa.citation_out_of_range", number=n, evidence_count=len(evidence))
            continue
        if n not in seen:
            seen.add(n)
            cited_pairs.append((n, evidence[n - 1].chunk_id))

    return answer.strip(), cited_pairs

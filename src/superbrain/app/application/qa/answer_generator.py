"""Grounded answer generation using retrieved evidence chunks."""

import re
from uuid import UUID

import structlog

from superbrain.app.application.ports import LLMPort
from superbrain.app.application.qa.evidence_builder import Evidence

log = structlog.get_logger(__name__)

GROUNDED_QA_PROMPT = """You are a question-answering assistant. Your job is to answer the question below using ONLY the evidence provided. You are not allowed to use any knowledge outside of the provided evidence.

QUESTION:
{question}

EVIDENCE:
{evidence_block}

RULES:
- Answer using ONLY information present in the evidence above
- If the evidence does not contain enough information to answer the question, say exactly: "I cannot answer this question based on the available evidence."
- Do not speculate, infer, or use outside knowledge
- Keep your answer concise — 2 to 5 sentences unless the question requires more detail
- After your answer, list the source IDs you used, in this exact format:
  SOURCES: chunk_id_1, chunk_id_2

You must always end your response with a SOURCES line, even if you could not answer.

Begin your answer now:"""


def format_evidence_block(evidence: list[Evidence]) -> str:
    lines = []
    for e in evidence:
        lines.append(f"[CHUNK {e.chunk_id}]")
        lines.append(f"Source: {e.article_title or e.article_url}")
        lines.append(e.content)
        lines.append("")
    return "\n".join(lines)


async def generate_answer(
    llm: LLMPort,
    model: str,
    question: str,
    evidence: list[Evidence],
) -> tuple[str, list[UUID]]:
    """Generate a grounded answer and return (answer_text, cited_chunk_ids)."""
    prompt = GROUNDED_QA_PROMPT.format(
        question=question,
        evidence_block=format_evidence_block(evidence),
    )
    raw = await llm.complete(prompt, model=model, prompt_template="grounded_qa_v1")
    return parse_answer_response(raw, evidence)


def parse_answer_response(
    raw: str, evidence: list[Evidence]
) -> tuple[str, list[UUID]]:
    """Split model output into answer text and cited chunk IDs.

    Defensive: local models sometimes omit or mangle the SOURCES line.
    Hallucinated IDs (not in the evidence set) are logged and dropped.
    """
    valid_ids = {str(e.chunk_id) for e in evidence}

    parts = re.split(r"\nSOURCES:\s*", raw, maxsplit=1)
    answer_text = parts[0].strip()

    if len(parts) < 2:
        log.warning("qa.missing_sources_line", raw=raw[:200])
        return answer_text, []

    cited_ids: list[UUID] = []
    for id_str in re.split(r"[,\s]+", parts[1].strip()):
        id_str = id_str.strip()
        if not id_str:
            continue
        if id_str in valid_ids:
            cited_ids.append(UUID(id_str))
        else:
            log.warning("qa.invalid_source_id", id=id_str)

    return answer_text, cited_ids

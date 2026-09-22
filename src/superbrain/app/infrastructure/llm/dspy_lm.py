"""dspy.LM adapter that routes every DSPy call through our own LLMPort.

Wrapping LLMPort — instead of pointing DSPy at Ollama directly via litellm — keeps
every DSPy call on the exact same path as every other LLM call in the app: the
retry/backoff in OllamaLLM, the model_call_logs audit trail, and PrioritizedLLM's
queue. DSPy never talks to Ollama itself.

forward() is synchronous (DSPy's contract) and must run off the main event loop —
callers invoke the surrounding dspy.Predict via asyncio.to_thread(), so this
asyncio.run() always starts in a thread with no event loop of its own to clash with.
"""

from __future__ import annotations

import asyncio
import re

import dspy

from superbrain.app.application.ports import LLMPort

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(raw: str) -> str:
    """Strip <think>...</think> reasoning blocks some local models emit inline."""
    cleaned = _THINK_RE.sub("", raw)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned


def _flatten(messages: list) -> str:
    """Join DSPy's system/user messages into the single flat prompt Ollama expects."""
    parts = ["".join(p.text for p in m.parts if hasattr(p, "text")) for m in messages]
    return "\n\n".join(parts)


class LLMPortLM(dspy.BaseLM):
    """A one-shot dspy.LM bound to a single LLMPort call at a fixed model/template.

    Cheap to construct — build a fresh instance per call (see analyze_query, rerank,
    generate_answer) rather than sharing one across requests.
    """

    forward_contract = "typed_lm"

    def __init__(self, llm: LLMPort, *, model: str, prompt_template: str) -> None:
        super().__init__(model=model, num_retries=0)  # LLMPort already retries internally
        self._llm = llm
        self._prompt_template = prompt_template
        self.last_prompt: str | None = None  # captured for the retrieval trace

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        prompt = _flatten(request.messages)
        self.last_prompt = prompt
        text = asyncio.run(
            self._llm.complete(
                prompt,
                model=self.model,
                json_mode=True,  # matches JSONAdapter's expected output shape
                prompt_template=self._prompt_template,
            )
        )
        return dspy.LMResponse.from_text(_strip_think(text), model=self.model)

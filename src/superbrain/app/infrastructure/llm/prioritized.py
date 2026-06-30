"""LLMPort wrapper that routes calls through a shared LLMPriorityQueue."""

from uuid import UUID

from superbrain.app.application.ports import LLMPort
from superbrain.app.infrastructure.llm.priority_queue import LLMPriorityQueue


class PrioritizedLLM(LLMPort):
    """Wraps any LLMPort and submits all calls at a fixed priority level.

    Use three instances backed by the same queue:
        qa_llm         = PrioritizedLLM(base, queue, priority=1)
        ingestion_llm  = PrioritizedLLM(base, queue, priority=2)
        background_llm = PrioritizedLLM(base, queue, priority=3)
    """

    def __init__(self, base: LLMPort, queue: LLMPriorityQueue, *, priority: int) -> None:
        self._base = base
        self._queue = queue
        self._priority = priority

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        json_mode: bool = False,
        prompt_template: str = "unknown",
        related_entity_id: UUID | None = None,
    ) -> str:
        return await self._queue.submit(
            lambda: self._base.complete(
                prompt,
                model=model,
                json_mode=json_mode,
                prompt_template=prompt_template,
                related_entity_id=related_entity_id,
            ),
            priority=self._priority,
        )

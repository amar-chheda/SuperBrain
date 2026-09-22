"""Regression test: LLMPortLM.forward() must not deadlock across event loops.

Reproduces the real production shape — a PrioritizedLLM whose LLMPriorityQueue
worker lives on the main loop, called from a to_thread worker (DSPy's sync
contract) — and asserts the call actually completes instead of hanging.
"""

import asyncio

import dspy
import pytest
from dspy.core.types import LMTextPart

from superbrain.app.application.ports import LLMPort
from superbrain.app.infrastructure.llm.dspy_lm import LLMPortLM
from superbrain.app.infrastructure.llm.prioritized import PrioritizedLLM
from superbrain.app.infrastructure.llm.priority_queue import LLMPriorityQueue


class _StubLLM(LLMPort):
    async def complete(self, prompt, *, model, json_mode=False,
                        prompt_template="unknown", related_entity_id=None) -> str:
        return '{"ok": true}'


@pytest.mark.asyncio
async def test_forward_completes_through_priority_queue_from_worker_thread():
    queue = LLMPriorityQueue()
    queue.start()  # worker task binds to *this* (the test's) running loop
    try:
        llm = PrioritizedLLM(_StubLLM(), queue, priority=1)
        stage_lm = LLMPortLM(llm, model="stub", prompt_template="test")

        request = dspy.LMRequest(
            model="stub",
            messages=[dspy.LMMessage(role="user", parts=[LMTextPart(text="hi")])],
        )
        # forward() is sync and must run off-loop, exactly as callers do it.
        response = await asyncio.wait_for(
            asyncio.to_thread(stage_lm.forward, request), timeout=5
        )
        assert response.outputs[0].parts[0].text == '{"ok": true}'
    finally:
        queue.stop()

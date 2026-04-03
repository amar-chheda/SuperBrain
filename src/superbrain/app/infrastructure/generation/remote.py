"""Remote chat generation providers for Ollama and LM Studio runtimes."""

from datetime import UTC, datetime

import httpx

from superbrain.app.application.qa.models import GeneratedAnswer
from superbrain.app.application.retrieval.models import EvidenceSet
from superbrain.app.observability.model_calls import ModelCallLogger, ModelCallPayload


class RemoteChatProvider:
    """HTTP-based answer provider with retries and health checks."""

    def __init__(
        self,
        *,
        runtime: str,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
        max_retries: int,
        model_call_logger: ModelCallLogger | None = None,
    ) -> None:
        self._runtime = runtime
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._model_call_logger = model_call_logger

    def generate_answer(self, question: str, evidence: EvidenceSet) -> GeneratedAnswer:
        started_at = datetime.now(UTC)
        if not evidence.chunks:
            self._log_call("generate_answer", started_at, "low_evidence", None)
            return GeneratedAnswer(
                answer="I do not have enough evidence in saved articles to answer reliably.",
                supported=False,
                citation_chunk_ids=[],
            )

        context = "\n\n".join(
            f"[{index + 1}] {chunk.chunk.chunk_text}"
            for index, chunk in enumerate(evidence.chunks[:6])
        )
        prompt = (
            "Answer the question using only the evidence below. "
            "If evidence is insufficient, say so.\n\n"
            f"Question: {question}\n\nEvidence:\n{context}"
        )

        try:
            if self._runtime == "ollama":
                text = self._ollama_generate(prompt)
            else:
                text = self._lmstudio_generate(prompt)

            citations = [str(chunk.chunk.chunk_id) for chunk in evidence.chunks[:2]]
            self._log_call("generate_answer", started_at, "success", None)
            return GeneratedAnswer(
                answer=text.strip(),
                supported=True,
                citation_chunk_ids=citations,
            )
        except Exception as exc:
            self._log_call("generate_answer", started_at, "failed", str(exc))
            raise

    def health_check(self) -> bool:
        try:
            if self._runtime == "ollama":
                response = self._request("GET", f"{self._base_url}/api/tags", None)
                return response.status_code == 200
            response = self._request("GET", f"{self._base_url}/v1/models", None)
            return response.status_code == 200
        except Exception:
            return False

    def _ollama_generate(self, prompt: str) -> str:
        response = self._request(
            "POST",
            f"{self._base_url}/api/generate",
            {"model": self._model_name, "prompt": prompt, "stream": False},
        )
        payload = response.json()
        return str(payload.get("response", ""))

    def _lmstudio_generate(self, prompt: str) -> str:
        response = self._request(
            "POST",
            f"{self._base_url}/v1/chat/completions",
            {
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        )
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])

    def _request(
        self, method: str, url: str, json_body: dict[str, object] | None
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.request(method, url, json=json_body)
                response.raise_for_status()
                return response
            except Exception as exc:
                last_error = exc
                if attempt == self._max_retries:
                    break
        assert last_error is not None
        raise last_error

    def _log_call(
        self,
        request_type: str,
        started_at: datetime,
        status: str,
        error_metadata: str | None,
    ) -> None:
        if self._model_call_logger is None:
            return
        self._model_call_logger.log(
            ModelCallPayload(
                provider=self._runtime,
                model_name=self._model_name,
                request_type=request_type,
                prompt_template="grounded_qa",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status=status,
                retries=self._max_retries,
                error_metadata=error_metadata,
            )
        )

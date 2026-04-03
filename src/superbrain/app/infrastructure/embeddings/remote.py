"""Remote embedding providers for Ollama and LM Studio runtimes."""

from datetime import UTC, datetime

import httpx

from superbrain.app.application.ports import EmbeddingProvider
from superbrain.app.observability.model_calls import ModelCallLogger, ModelCallPayload


class RemoteEmbeddingProvider(EmbeddingProvider):
    """HTTP-based embedding provider with retry/backoff and health checks."""

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

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started_at = datetime.now(UTC)
        try:
            if self._runtime == "ollama":
                vectors = [self._ollama_embed(text) for text in texts]
            else:
                vectors = self._lmstudio_embed_batch(texts)
            self._log_call("embed_documents", started_at, "success", None)
            return vectors
        except Exception as exc:
            self._log_call("embed_documents", started_at, "failed", str(exc))
            raise

    def embed_query(self, text: str) -> list[float]:
        started_at = datetime.now(UTC)
        try:
            if self._runtime == "ollama":
                vector = self._ollama_embed(text)
            else:
                vector = self._lmstudio_embed_batch([text])[0]
            self._log_call("embed_query", started_at, "success", None)
            return vector
        except Exception as exc:
            self._log_call("embed_query", started_at, "failed", str(exc))
            raise

    def health_check(self) -> bool:
        try:
            if self._runtime == "ollama":
                response = self._request("GET", f"{self._base_url}/api/tags", json_body=None)
                return response.status_code == 200
            response = self._request("GET", f"{self._base_url}/v1/models", json_body=None)
            return response.status_code == 200
        except Exception:
            return False

    def _ollama_embed(self, text: str) -> list[float]:
        response = self._request(
            "POST",
            f"{self._base_url}/api/embeddings",
            json_body={"model": self._model_name, "prompt": text},
        )
        payload = response.json()
        return [float(value) for value in payload["embedding"]]

    def _lmstudio_embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._request(
            "POST",
            f"{self._base_url}/v1/embeddings",
            json_body={"model": self._model_name, "input": texts},
        )
        payload = response.json()
        rows = payload["data"]
        return [[float(value) for value in row["embedding"]] for row in rows]

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
                prompt_template=None,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status=status,
                retries=self._max_retries,
                error_metadata=error_metadata,
            )
        )

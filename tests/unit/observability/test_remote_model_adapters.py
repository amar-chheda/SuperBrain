"""Unit tests for remote model runtime adapters."""

from types import MethodType

from superbrain.app.application.retrieval.models import EvidenceSet
from superbrain.app.infrastructure.embeddings.remote import RemoteEmbeddingProvider
from superbrain.app.infrastructure.generation.remote import RemoteChatProvider


def test_remote_embedding_provider_health_check_failure() -> None:
    """Health check should return false when endpoint call fails."""

    provider = RemoteEmbeddingProvider(
        runtime="ollama",
        base_url="http://localhost:11434",
        model_name="nomic-embed-text",
        timeout_seconds=1,
        max_retries=1,
    )

    def fail_request(self, method: str, url: str, json_body):
        raise RuntimeError("unreachable")

    provider._request = MethodType(fail_request, provider)  # type: ignore[method-assign]
    assert provider.health_check() is False


def test_remote_chat_provider_low_evidence_short_circuit() -> None:
    """Remote chat adapter should refuse without evidence before remote call."""

    provider = RemoteChatProvider(
        runtime="lmstudio",
        base_url="http://localhost:1234",
        model_name="llama",
        timeout_seconds=1,
        max_retries=1,
    )

    result = provider.generate_answer(question="q", evidence=EvidenceSet(chunks=tuple()))

    assert result.supported is False
    assert "enough evidence" in result.answer.lower()

"""Local topic classifier implementation."""

import re
from datetime import UTC, datetime

from superbrain.app.application.topics.classification import TopicClassifier
from superbrain.app.application.topics.models import TopicClassificationDecision, TopicWithVersion
from superbrain.app.observability.model_calls import ModelCallLogger, ModelCallPayload


class LocalKeywordTopicClassifier(TopicClassifier):
    """Heuristic classifier with rationale/disqualifier metadata."""

    def __init__(self, model_call_logger: ModelCallLogger | None = None) -> None:
        self._model_call_logger = model_call_logger

    def classify(
        self,
        article_text: str,
        topics: list[TopicWithVersion],
    ) -> list[TopicClassificationDecision]:
        """Classify article text against topic definitions using keyword overlap."""

        started_at = datetime.now(UTC)
        tokens = _tokenize(article_text)
        decisions: list[TopicClassificationDecision] = []

        for candidate in topics:
            positive_tokens = _tokenize(
                " ".join((candidate.version.description, *candidate.version.positive_examples))
            )
            negative_tokens = _tokenize(" ".join(candidate.version.negative_examples))

            positive_hits = sorted(tokens.intersection(positive_tokens))
            negative_hits = sorted(tokens.intersection(negative_tokens))

            positive_score = len(positive_hits) / max(1, len(positive_tokens))
            negative_penalty = len(negative_hits) / max(1, len(negative_tokens))
            priority_boost = 1 / max(1, candidate.topic.priority)
            score = max(0.0, min(1.0, positive_score - (0.7 * negative_penalty) + priority_boost))

            matched = score >= 0.1 and (len(negative_hits) <= len(positive_hits))
            rationale = (
                f"positive_hits={positive_hits[:5]}, "
                f"negative_hits={negative_hits[:5]}, score={score:.3f}"
            )

            decisions.append(
                TopicClassificationDecision(
                    topic_id=candidate.topic.id,
                    topic_version_id=candidate.version.id,
                    matched=matched,
                    score=score,
                    rationale=rationale,
                    disqualifiers=tuple(negative_hits[:5]),
                )
            )

        decisions.sort(key=lambda item: item.score, reverse=True)
        self._log_call(started_at=started_at, status="success")
        return decisions

    def _log_call(self, *, started_at: datetime, status: str) -> None:
        if self._model_call_logger is None:
            return
        self._model_call_logger.log(
            ModelCallPayload(
                provider="local_keyword",
                model_name="keyword_classifier",
                request_type="classify_topic",
                prompt_template="topic_classifier_v1",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                status=status,
            )
        )


def _tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", value.lower())}

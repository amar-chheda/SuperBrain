"""Dependency providers for FastAPI routes."""

from collections.abc import Generator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from superbrain.app.application.digest.deduplication import CanonicalUrlDigestDeduper, DigestDeduper
from superbrain.app.application.digest.generator import DigestGenerator
from superbrain.app.application.digest.use_case import GenerateDailyDigestUseCase
from superbrain.app.application.ingestion.chunking import ParagraphChunkingStrategy
from superbrain.app.application.ingestion.retry import RetryFailedIngestionUseCase
from superbrain.app.application.ingestion.url import DefaultUrlCanonicalizer
from superbrain.app.application.ingestion.use_case import IngestUrlUseCase
from superbrain.app.application.ports import (
    ArticleExtractor,
    ChatModelProvider,
    ChunkingStrategy,
    EmbeddingProvider,
    RetrievalService,
    UrlCanonicalizer,
)
from superbrain.app.application.qa.citations import CitationBuilder
from superbrain.app.application.qa.use_case import AskQuestionUseCase
from superbrain.app.application.retrieval.service import HybridRetrievalService
from superbrain.app.application.topics.classification import (
    ClassifyArticleUseCase,
    ReclassifyArticlesUseCase,
    TopicClassifier,
)
from superbrain.app.application.topics.service import TopicService
from superbrain.app.config.settings import AppSettings, get_settings
from superbrain.app.domain.repositories import (
    ArticleRepository,
    ArticleTopicMatchRepository,
    DigestRepository,
    IngestionJobRepository,
    ModelCallLogRepository,
    QueryLogRepository,
    RetrievalRepository,
    TopicRepository,
)
from superbrain.app.infrastructure.classification.local import LocalKeywordTopicClassifier
from superbrain.app.infrastructure.db.session import get_session_factory
from superbrain.app.infrastructure.embeddings.local import LocalHashEmbeddingProvider
from superbrain.app.infrastructure.embeddings.remote import RemoteEmbeddingProvider
from superbrain.app.infrastructure.extractors.local import (
    ChainedArticleExtractor,
    FallbackArticleExtractor,
    HttpArticleExtractor,
)
from superbrain.app.infrastructure.generation.local import LocalGroundedChatProvider
from superbrain.app.infrastructure.generation.remote import RemoteChatProvider
from superbrain.app.infrastructure.notifiers.local import LoggingTelegramNotifier
from superbrain.app.infrastructure.repositories.sqlalchemy import (
    SqlAlchemyArticleRepository,
    SqlAlchemyArticleTopicMatchRepository,
    SqlAlchemyDigestRepository,
    SqlAlchemyIngestionJobRepository,
    SqlAlchemyModelCallLogRepository,
    SqlAlchemyQueryLogRepository,
    SqlAlchemyRetrievalRepository,
    SqlAlchemyTopicRepository,
)
from superbrain.app.infrastructure.scheduling.persistent import PersistentScheduler
from superbrain.app.observability.metrics import (
    InMemoryMetricsRecorder,
    MetricsRecorder,
    PrometheusMetricsRecorder,
)
from superbrain.app.observability.model_calls import ModelCallLogger


def get_app_settings() -> AppSettings:
    return get_settings()


def get_db_session() -> Generator[Session, None, None]:
    session_factory = get_session_factory()
    with session_factory() as session:
        yield session


def get_url_canonicalizer() -> UrlCanonicalizer:
    return DefaultUrlCanonicalizer()


def get_article_extractor() -> ArticleExtractor:
    return ChainedArticleExtractor(
        primary=HttpArticleExtractor(),
        fallback=FallbackArticleExtractor(),
    )


def get_chunking_strategy() -> ChunkingStrategy:
    return ParagraphChunkingStrategy()


def get_model_call_log_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> ModelCallLogRepository:
    return SqlAlchemyModelCallLogRepository(session=session)


def get_model_call_logger(
    repository: Annotated[ModelCallLogRepository, Depends(get_model_call_log_repository)],
) -> ModelCallLogger:
    return ModelCallLogger(repository=repository)


def get_embedding_provider(
    settings: Annotated[AppSettings, Depends(get_app_settings)],
    model_call_logger: Annotated[ModelCallLogger, Depends(get_model_call_logger)],
) -> EmbeddingProvider:
    if settings.model_runtime in {"ollama", "lmstudio"}:
        return RemoteEmbeddingProvider(
            runtime=settings.model_runtime,
            base_url=settings.local_model_base_url,
            model_name=settings.embedding_model_name,
            timeout_seconds=settings.model_request_timeout_seconds,
            max_retries=settings.model_max_retries,
            model_call_logger=model_call_logger,
        )
    return LocalHashEmbeddingProvider(
        dimensions=settings.embedding_dimensions,
        model_call_logger=model_call_logger,
    )


def get_chat_model_provider(
    settings: Annotated[AppSettings, Depends(get_app_settings)],
    model_call_logger: Annotated[ModelCallLogger, Depends(get_model_call_logger)],
) -> ChatModelProvider:
    if settings.model_runtime in {"ollama", "lmstudio"}:
        return RemoteChatProvider(
            runtime=settings.model_runtime,
            base_url=settings.local_model_base_url,
            model_name=settings.generation_model_name,
            timeout_seconds=settings.model_request_timeout_seconds,
            max_retries=settings.model_max_retries,
            model_call_logger=model_call_logger,
        )
    return LocalGroundedChatProvider(model_call_logger=model_call_logger)


def get_topic_classifier(
    model_call_logger: Annotated[ModelCallLogger, Depends(get_model_call_logger)],
) -> TopicClassifier:
    return LocalKeywordTopicClassifier(model_call_logger=model_call_logger)


def get_digest_notifier() -> LoggingTelegramNotifier:
    return LoggingTelegramNotifier()


def get_digest_deduper() -> DigestDeduper:
    return CanonicalUrlDigestDeduper()


def get_digest_generator() -> DigestGenerator:
    return DigestGenerator()


def get_scheduler() -> PersistentScheduler:
    return PersistentScheduler(session_factory=get_session_factory())


@lru_cache(maxsize=1)
def get_metrics_recorder() -> MetricsRecorder:
    settings = get_settings()
    if settings.metrics_backend == "prometheus":
        return PrometheusMetricsRecorder()
    return InMemoryMetricsRecorder()


def get_article_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> ArticleRepository:
    return SqlAlchemyArticleRepository(session=session)


def get_article_topic_match_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> ArticleTopicMatchRepository:
    return SqlAlchemyArticleTopicMatchRepository(session=session)


def get_topic_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> TopicRepository:
    return SqlAlchemyTopicRepository(session=session)


def get_digest_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> DigestRepository:
    return SqlAlchemyDigestRepository(session=session)


def get_retrieval_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> RetrievalRepository:
    return SqlAlchemyRetrievalRepository(session=session)


def get_query_log_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> QueryLogRepository:
    return SqlAlchemyQueryLogRepository(session=session)


def get_ingestion_job_repository(
    session: Annotated[Session, Depends(get_db_session)],
) -> IngestionJobRepository:
    return SqlAlchemyIngestionJobRepository(session=session)


def get_ingest_url_use_case(
    article_repository: Annotated[ArticleRepository, Depends(get_article_repository)],
    ingestion_job_repository: Annotated[
        IngestionJobRepository,
        Depends(get_ingestion_job_repository),
    ],
    canonicalizer: Annotated[UrlCanonicalizer, Depends(get_url_canonicalizer)],
    extractor: Annotated[ArticleExtractor, Depends(get_article_extractor)],
    chunking_strategy: Annotated[ChunkingStrategy, Depends(get_chunking_strategy)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    metrics: Annotated[MetricsRecorder, Depends(get_metrics_recorder)],
) -> IngestUrlUseCase:
    return IngestUrlUseCase(
        article_repository=article_repository,
        ingestion_job_repository=ingestion_job_repository,
        canonicalizer=canonicalizer,
        extractor=extractor,
        chunking_strategy=chunking_strategy,
        embedding_provider=embedding_provider,
        metrics=metrics,
    )


def get_retrieval_service(
    retrieval_repository: Annotated[RetrievalRepository, Depends(get_retrieval_repository)],
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> RetrievalService:
    return HybridRetrievalService(
        retrieval_repository=retrieval_repository,
        embedding_provider=embedding_provider,
    )


def get_ask_question_use_case(
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    chat_provider: Annotated[ChatModelProvider, Depends(get_chat_model_provider)],
    query_log_repository: Annotated[QueryLogRepository, Depends(get_query_log_repository)],
    metrics: Annotated[MetricsRecorder, Depends(get_metrics_recorder)],
) -> AskQuestionUseCase:
    return AskQuestionUseCase(
        retrieval_service=retrieval_service,
        chat_provider=chat_provider,
        citation_builder=CitationBuilder(),
        query_log_repository=query_log_repository,
        metrics=metrics,
    )


def get_topic_service(
    topic_repository: Annotated[TopicRepository, Depends(get_topic_repository)],
) -> TopicService:
    return TopicService(topic_repository=topic_repository)


def get_classify_article_use_case(
    article_repository: Annotated[ArticleRepository, Depends(get_article_repository)],
    topic_repository: Annotated[TopicRepository, Depends(get_topic_repository)],
    match_repository: Annotated[
        ArticleTopicMatchRepository,
        Depends(get_article_topic_match_repository),
    ],
    classifier: Annotated[TopicClassifier, Depends(get_topic_classifier)],
    metrics: Annotated[MetricsRecorder, Depends(get_metrics_recorder)],
) -> ClassifyArticleUseCase:
    return ClassifyArticleUseCase(
        article_repository=article_repository,
        topic_repository=topic_repository,
        match_repository=match_repository,
        classifier=classifier,
        metrics=metrics,
    )


def get_reclassify_articles_use_case(
    article_repository: Annotated[ArticleRepository, Depends(get_article_repository)],
    classify_article_use_case: Annotated[
        ClassifyArticleUseCase,
        Depends(get_classify_article_use_case),
    ],
) -> ReclassifyArticlesUseCase:
    return ReclassifyArticlesUseCase(
        article_repository=article_repository,
        classify_article_use_case=classify_article_use_case,
    )


def get_retry_failed_ingestion_use_case(
    ingestion_job_repository: Annotated[
        IngestionJobRepository,
        Depends(get_ingestion_job_repository),
    ],
    ingest_url_use_case: Annotated[IngestUrlUseCase, Depends(get_ingest_url_use_case)],
) -> RetryFailedIngestionUseCase:
    return RetryFailedIngestionUseCase(
        ingestion_job_repository=ingestion_job_repository,
        ingest_url_use_case=ingest_url_use_case,
    )


def get_generate_daily_digest_use_case(
    article_repository: Annotated[ArticleRepository, Depends(get_article_repository)],
    topic_repository: Annotated[TopicRepository, Depends(get_topic_repository)],
    match_repository: Annotated[
        ArticleTopicMatchRepository,
        Depends(get_article_topic_match_repository),
    ],
    digest_repository: Annotated[DigestRepository, Depends(get_digest_repository)],
    deduper: Annotated[DigestDeduper, Depends(get_digest_deduper)],
    generator: Annotated[DigestGenerator, Depends(get_digest_generator)],
    notifier: Annotated[LoggingTelegramNotifier, Depends(get_digest_notifier)],
    metrics: Annotated[MetricsRecorder, Depends(get_metrics_recorder)],
) -> GenerateDailyDigestUseCase:
    return GenerateDailyDigestUseCase(
        article_repository=article_repository,
        topic_repository=topic_repository,
        match_repository=match_repository,
        digest_repository=digest_repository,
        deduper=deduper,
        generator=generator,
        notifier=notifier,
        metrics=metrics,
    )

"""SQLAlchemy-backed repository implementations."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from superbrain.app.domain.models import (
    Article,
    ArticleChunk,
    Digest,
    DigestItem,
    DigestStatus,
    IngestionJob,
    IngestionStatus,
    QueryLogEntry,
    StoredChunk,
    TopicDefinition,
    TopicMatch,
    TopicStatus,
    TopicVersion,
)
from superbrain.app.infrastructure.db.models import (
    ArticleChunkRecord,
    ArticleRawSnapshotRecord,
    ArticleRecord,
    ArticleTopicMatchRecord,
    DigestItemRecord,
    DigestRunRecord,
    IngestionJobRecord,
    ModelCallLogRecord,
    QueryLogRecord,
    TopicRecord,
    TopicVersionRecord,
)


class SqlAlchemyArticleRepository:
    """Persist articles, snapshots, and chunks using SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, article: Article) -> Article:
        record = ArticleRecord(
            id=article.id,
            source_url=article.source_url,
            canonical_url=article.canonical_url,
            domain=article.domain,
            title=article.title,
            author=article.author,
            published_at=article.published_at,
            content=article.content,
            content_hash=article.content_hash,
            extraction_quality_score=article.extraction_quality_score,
            extraction_notes=article.extraction_notes,
            created_at=article.created_at,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return _to_article(record)

    def save_chunks(self, chunks: list[ArticleChunk]) -> list[ArticleChunk]:
        records = [
            ArticleChunkRecord(
                id=chunk.id,
                article_id=chunk.article_id,
                chunk_index=chunk.index,
                text=chunk.text,
                token_count=chunk.token_count,
                embedding=chunk.embedding,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
            )
            for chunk in chunks
        ]
        self._session.add_all(records)
        self._session.commit()
        for record in records:
            self._session.refresh(record)
        return [_to_chunk(record) for record in records]

    def save_raw_snapshot(self, article_id: UUID, raw_html: str) -> None:
        self._session.add(ArticleRawSnapshotRecord(article_id=article_id, raw_html=raw_html))
        self._session.commit()

    def get(self, article_id: UUID) -> Article | None:
        record = self._session.get(ArticleRecord, article_id)
        return _to_article(record) if record is not None else None

    def list_articles(
        self,
        limit: int = 100,
        article_ids: list[UUID] | None = None,
    ) -> list[Article]:
        stmt = select(ArticleRecord).order_by(ArticleRecord.created_at.desc()).limit(limit)
        if article_ids is not None:
            stmt = stmt.where(ArticleRecord.id.in_(article_ids))
        records = self._session.scalars(stmt).all()
        return [_to_article(record) for record in records]

    def list_between(self, start: datetime, end: datetime) -> list[Article]:
        stmt = select(ArticleRecord).where(
            ArticleRecord.created_at >= start,
            ArticleRecord.created_at < end,
        )
        records = self._session.scalars(stmt).all()
        return [_to_article(record) for record in records]

    def get_by_source_url(self, source_url: str) -> Article | None:
        record = self._session.scalar(
            select(ArticleRecord).where(ArticleRecord.source_url == source_url)
        )
        return _to_article(record) if record is not None else None

    def get_by_canonical_url(self, canonical_url: str) -> Article | None:
        record = self._session.scalar(
            select(ArticleRecord).where(ArticleRecord.canonical_url == canonical_url)
        )
        return _to_article(record) if record is not None else None

    def get_by_content_hash(self, content_hash: str) -> Article | None:
        record = self._session.scalar(
            select(ArticleRecord).where(ArticleRecord.content_hash == content_hash)
        )
        return _to_article(record) if record is not None else None


class SqlAlchemyArticleTopicMatchRepository:
    """Persist article-topic classification matches."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_for_article(self, article_id: UUID, matches: list[TopicMatch]) -> list[TopicMatch]:
        self._session.execute(
            delete(ArticleTopicMatchRecord).where(ArticleTopicMatchRecord.article_id == article_id)
        )

        records = [
            ArticleTopicMatchRecord(
                id=uuid4(),
                article_id=match.article_id,
                topic_id=match.topic_id,
                topic_version_id=match.topic_version_id,
                score=match.score,
                rationale=match.rationale,
                disqualifiers=list(match.disqualifiers),
                classified_at=match.classified_at,
            )
            for match in matches
        ]
        if records:
            self._session.add_all(records)
        self._session.commit()
        return [
            TopicMatch(
                article_id=record.article_id,
                topic_id=record.topic_id,
                topic_version_id=record.topic_version_id,
                score=record.score,
                rationale=record.rationale,
                disqualifiers=tuple(record.disqualifiers),
                classified_at=record.classified_at,
            )
            for record in records
        ]

    def list_for_article(self, article_id: UUID) -> list[TopicMatch]:
        records = self._session.scalars(
            select(ArticleTopicMatchRecord).where(ArticleTopicMatchRecord.article_id == article_id)
        ).all()
        return [
            TopicMatch(
                article_id=record.article_id,
                topic_id=record.topic_id,
                topic_version_id=record.topic_version_id,
                score=record.score,
                rationale=record.rationale,
                disqualifiers=tuple(record.disqualifiers),
                classified_at=record.classified_at,
            )
            for record in records
        ]

    def list_for_articles(self, article_ids: list[UUID]) -> list[TopicMatch]:
        if not article_ids:
            return []
        records = self._session.scalars(
            select(ArticleTopicMatchRecord).where(ArticleTopicMatchRecord.article_id.in_(article_ids))
        ).all()
        return [
            TopicMatch(
                article_id=record.article_id,
                topic_id=record.topic_id,
                topic_version_id=record.topic_version_id,
                score=record.score,
                rationale=record.rationale,
                disqualifiers=tuple(record.disqualifiers),
                classified_at=record.classified_at,
            )
            for record in records
        ]


class SqlAlchemyTopicRepository:
    """Persist topic metadata and version definitions."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, topic: TopicDefinition, version: TopicVersion) -> TopicDefinition:
        self._session.add(_from_topic(topic))
        self._session.add(_from_topic_version(version))
        self._session.commit()
        return topic

    def update(self, topic: TopicDefinition, version: TopicVersion) -> TopicDefinition:
        topic_record = self._session.get(TopicRecord, topic.id)
        if topic_record is None:
            raise ValueError("topic not found")

        topic_record.status = topic.status.value
        topic_record.priority = topic.priority
        topic_record.current_version_id = topic.current_version_id
        topic_record.updated_at = topic.updated_at

        self._session.add(topic_record)
        self._session.add(_from_topic_version(version))
        self._session.commit()
        return _to_topic(topic_record)

    def set_inactive(self, topic_id: UUID) -> TopicDefinition:
        record = self._session.get(TopicRecord, topic_id)
        if record is None:
            raise ValueError("topic not found")
        record.status = TopicStatus.INACTIVE.value
        record.updated_at = datetime.now(UTC)
        self._session.add(record)
        self._session.commit()
        return _to_topic(record)

    def get(self, topic_id: UUID) -> TopicDefinition | None:
        record = self._session.get(TopicRecord, topic_id)
        return _to_topic(record) if record is not None else None

    def list_all(self, active_only: bool = False) -> list[TopicDefinition]:
        stmt = select(TopicRecord).order_by(
            TopicRecord.priority.asc(),
            TopicRecord.created_at.asc(),
        )
        if active_only:
            stmt = stmt.where(TopicRecord.status == TopicStatus.ACTIVE.value)
        records = self._session.scalars(stmt).all()
        return [_to_topic(record) for record in records]

    def get_latest_version(self, topic_id: UUID) -> TopicVersion | None:
        topic = self._session.get(TopicRecord, topic_id)
        if topic is None:
            return None
        version = self._session.get(TopicVersionRecord, topic.current_version_id)
        return _to_topic_version(version) if version is not None else None

    def list_active_with_latest_versions(self) -> list[tuple[TopicDefinition, TopicVersion]]:
        topics = self.list_all(active_only=True)
        results: list[tuple[TopicDefinition, TopicVersion]] = []
        for topic in topics:
            version = self._session.get(TopicVersionRecord, topic.current_version_id)
            if version is None:
                continue
            results.append((topic, _to_topic_version(version)))
        return results


class SqlAlchemyRetrievalRepository:
    """Read retrieval candidates from persisted chunk/article records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def list_chunks(self, limit: int = 1000) -> list[StoredChunk]:
        stmt = (
            select(ArticleChunkRecord, ArticleRecord)
            .join(ArticleRecord, ArticleRecord.id == ArticleChunkRecord.article_id)
            .order_by(ArticleChunkRecord.created_at.desc())
            .limit(limit)
        )
        rows = self._session.execute(stmt).all()
        return [
            StoredChunk(
                chunk_id=chunk.id,
                article_id=article.id,
                article_title=article.title,
                article_url=article.source_url,
                chunk_text=chunk.text,
                embedding=chunk.embedding,
            )
            for chunk, article in rows
        ]

    def lexical_scores(self, query: str, limit: int = 200) -> dict[str, float]:
        dialect_name = self._session.get_bind().dialect.name
        if dialect_name != "postgresql":
            return {}

        statement = text(
            """
            SELECT
              ac.id::text AS chunk_id,
              ts_rank_cd(
                to_tsvector('english', ac.text),
                plainto_tsquery('english', :query)
              ) AS score
            FROM article_chunks ac
            WHERE to_tsvector('english', ac.text) @@ plainto_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :limit
            """
        )
        rows = self._session.execute(statement, {"query": query, "limit": limit}).all()
        return {row.chunk_id: float(row.score) for row in rows}


class SqlAlchemyIngestionJobRepository:
    """Persist ingestion job state using SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, job: IngestionJob) -> IngestionJob:
        record = IngestionJobRecord(
            id=job.id,
            source_url=job.source_url,
            canonical_url=job.canonical_url,
            status=job.status.value,
            requested_at=job.requested_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error_message=job.error_message,
            article_id=job.article_id,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return _to_job(record)

    def update_status(
        self,
        job_id: UUID,
        status: IngestionStatus,
        *,
        error_message: str | None = None,
        article_id: UUID | None = None,
    ) -> IngestionJob:
        record = self._session.get(IngestionJobRecord, job_id)
        if record is None:
            raise ValueError(f"ingestion job not found: {job_id}")

        now = datetime.now(UTC)
        if status == IngestionStatus.RUNNING and record.started_at is None:
            record.started_at = now
        if status in (IngestionStatus.SUCCEEDED, IngestionStatus.FAILED):
            record.finished_at = now

        record.status = status.value
        record.error_message = error_message
        if article_id is not None:
            record.article_id = article_id

        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return _to_job(record)

    def get(self, job_id: UUID) -> IngestionJob | None:
        record = self._session.get(IngestionJobRecord, job_id)
        return _to_job(record) if record is not None else None

    def list_failed(self, limit: int = 50) -> list[IngestionJob]:
        records = self._session.scalars(
            select(IngestionJobRecord)
            .where(IngestionJobRecord.status == IngestionStatus.FAILED.value)
            .order_by(IngestionJobRecord.finished_at.desc())
            .limit(limit)
        ).all()
        return [_to_job(record) for record in records]


class SqlAlchemyQueryLogRepository:
    """Persist query execution logs for observability."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def record(self, entry: QueryLogEntry) -> None:
        self._session.add(
            QueryLogRecord(
                request_id=entry.query_request.id,
                question=entry.query_request.question,
                answer=entry.query_response.answer,
                supported=entry.query_response.supported,
                retrieval_ms=entry.retrieval_ms,
                generation_ms=entry.generation_ms,
                evidence_chunk_ids=[str(chunk_id) for chunk_id in entry.evidence_chunk_ids],
                citations=[
                    {
                        "chunk_id": str(citation.chunk_id),
                        "article_id": str(citation.article_id),
                        "article_url": citation.article_url,
                    }
                    for citation in entry.query_response.citations
                ],
            )
        )
        self._session.commit()


class SqlAlchemyModelCallLogRepository:
    """Persist model-call audit entries."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def record(
        self,
        *,
        provider: str,
        model_name: str,
        request_type: str,
        prompt_template: str | None,
        started_at: datetime,
        finished_at: datetime,
        duration_ms: float,
        status: str,
        retries: int,
        error_metadata: str | None,
        related_entity_id: str | None,
    ) -> None:
        self._session.add(
            ModelCallLogRecord(
                provider=provider,
                model_name=model_name,
                request_type=request_type,
                prompt_template=prompt_template,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                status=status,
                retries=retries,
                error_metadata=error_metadata,
                related_entity_id=related_entity_id,
            )
        )
        self._session.commit()


class SqlAlchemyDigestRepository:
    """Persist digest runs and sections."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_run(self, run_date: datetime) -> Digest:
        digest_id = uuid4()
        created_at = datetime.now(UTC)
        self._session.add(
            DigestRunRecord(
                id=digest_id,
                run_date=run_date,
                status=DigestStatus.RUNNING.value,
                created_at=created_at,
                finished_at=None,
                error_message=None,
            )
        )
        self._session.commit()
        return Digest(
            id=digest_id,
            run_date=run_date,
            status=DigestStatus.RUNNING,
            created_at=created_at,
            items=tuple(),
        )

    def complete_run(self, digest: Digest) -> Digest:
        run_record = self._session.get(DigestRunRecord, digest.id)
        if run_record is None:
            raise ValueError("digest run not found")

        self._session.execute(
            delete(DigestItemRecord).where(DigestItemRecord.digest_run_id == digest.id)
        )
        for item in digest.items:
            self._session.add(
                DigestItemRecord(
                    id=uuid4(),
                    digest_run_id=digest.id,
                    topic_id=item.topic_id,
                    topic_name=item.topic_name,
                    summary=item.summary,
                    source_urls=list(item.source_urls),
                    citation_article_ids=[
                        str(article_id) for article_id in item.citation_article_ids
                    ],
                )
            )

        run_record.status = DigestStatus.SUCCEEDED.value
        run_record.finished_at = datetime.now(UTC)
        run_record.error_message = None
        self._session.add(run_record)
        self._session.commit()
        return self._load_digest(run_record.id)

    def fail_run(self, digest_id: UUID, error_message: str) -> Digest:
        run_record = self._session.get(DigestRunRecord, digest_id)
        if run_record is None:
            raise ValueError("digest run not found")
        run_record.status = DigestStatus.FAILED.value
        run_record.finished_at = datetime.now(UTC)
        run_record.error_message = error_message
        self._session.add(run_record)
        self._session.commit()
        return self._load_digest(run_record.id)

    def get_latest(self) -> Digest | None:
        run_record = self._session.scalar(
            select(DigestRunRecord).order_by(DigestRunRecord.created_at.desc()).limit(1)
        )
        if run_record is None:
            return None
        return self._load_digest(run_record.id)

    def list_recent(self, limit: int = 20) -> list[Digest]:
        records = self._session.scalars(
            select(DigestRunRecord).order_by(DigestRunRecord.created_at.desc()).limit(limit)
        ).all()
        return [self._load_digest(record.id) for record in records]

    def _load_digest(self, digest_id: UUID) -> Digest:
        run_record = self._session.get(DigestRunRecord, digest_id)
        if run_record is None:
            raise ValueError("digest run not found")

        item_records = self._session.scalars(
            select(DigestItemRecord).where(DigestItemRecord.digest_run_id == digest_id)
        ).all()
        items = tuple(
            DigestItem(
                topic_id=item.topic_id,
                topic_name=item.topic_name,
                summary=item.summary,
                source_urls=tuple(item.source_urls),
                citation_article_ids=tuple(UUID(value) for value in item.citation_article_ids),
            )
            for item in item_records
        )
        return Digest(
            id=run_record.id,
            run_date=run_record.run_date,
            status=DigestStatus(run_record.status),
            created_at=run_record.created_at,
            items=items,
        )


def _to_article(record: ArticleRecord) -> Article:
    return Article(
        id=record.id,
        source_url=record.source_url,
        canonical_url=record.canonical_url,
        domain=record.domain,
        title=record.title,
        author=record.author,
        published_at=record.published_at,
        content=record.content,
        content_hash=record.content_hash,
        extraction_quality_score=record.extraction_quality_score,
        extraction_notes=record.extraction_notes,
        created_at=record.created_at,
    )


def _to_chunk(record: ArticleChunkRecord) -> ArticleChunk:
    return ArticleChunk(
        id=record.id,
        article_id=record.article_id,
        index=record.chunk_index,
        text=record.text,
        token_count=record.token_count,
        embedding=record.embedding,
        char_start=record.char_start,
        char_end=record.char_end,
    )


def _to_job(record: IngestionJobRecord) -> IngestionJob:
    return IngestionJob(
        id=record.id,
        source_url=record.source_url,
        canonical_url=record.canonical_url,
        status=IngestionStatus(record.status),
        requested_at=record.requested_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error_message=record.error_message,
        article_id=record.article_id,
    )


def _to_topic(record: TopicRecord) -> TopicDefinition:
    return TopicDefinition(
        id=record.id,
        name=record.name,
        status=TopicStatus(record.status),
        priority=record.priority,
        current_version_id=record.current_version_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _from_topic(topic: TopicDefinition) -> TopicRecord:
    return TopicRecord(
        id=topic.id,
        name=topic.name,
        status=topic.status.value,
        priority=topic.priority,
        current_version_id=topic.current_version_id,
        created_at=topic.created_at,
        updated_at=topic.updated_at,
    )


def _to_topic_version(record: TopicVersionRecord) -> TopicVersion:
    return TopicVersion(
        id=record.id,
        topic_id=record.topic_id,
        version=record.version_number,
        description=record.description,
        positive_examples=tuple(record.positive_examples),
        negative_examples=tuple(record.negative_examples),
        created_at=record.created_at,
    )


def _from_topic_version(version: TopicVersion) -> TopicVersionRecord:
    return TopicVersionRecord(
        id=version.id,
        topic_id=version.topic_id,
        version_number=version.version,
        description=version.description,
        positive_examples=list(version.positive_examples),
        negative_examples=list(version.negative_examples),
        created_at=version.created_at,
    )

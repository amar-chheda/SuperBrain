"""Custom SQLAlchemy types used across persistence models."""

from sqlalchemy import JSON, TypeDecorator
from sqlalchemy.engine import Dialect
from sqlalchemy.sql.type_api import TypeEngine


class EmbeddingVector(TypeDecorator[list[float]]):
    """Use pgvector on Postgres and JSON on other dialects for tests."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int) -> None:
        """Initialize vector dimension metadata."""

        super().__init__()
        self._dimensions = dimensions

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[object]:
        """Select dialect-specific storage type for embeddings."""

        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]

            return dialect.type_descriptor(Vector(self._dimensions))
        return dialect.type_descriptor(JSON())

    def process_bind_param(
        self,
        value: list[float] | None,
        dialect: Dialect,
    ) -> list[float] | None:
        """Validate and serialize vectors when writing rows."""

        _ = dialect
        if value is None:
            return None
        return [float(item) for item in value]

    def process_result_value(
        self,
        value: list[float] | None,
        dialect: Dialect,
    ) -> list[float] | None:
        """Normalize vectors when reading rows."""

        _ = dialect
        if value is None:
            return None
        return [float(item) for item in value]

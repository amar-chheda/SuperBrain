"""One-time admin job: re-embed every chunk with the nomic `search_document:` prefix.

Why this exists
---------------
Chunks were originally embedded with NO nomic task prefix. nomic-embed-text is
trained to require `search_document:` (corpus) and `search_query:` (queries);
without them, query/document cosine similarities are systematically depressed and
genuine matches fall under the retrieval floor. After the prefix fix, the *queries*
use `search_query:` but the *stored* vectors are still prefix-less — a mismatched
space. This script recomputes every chunk embedding so both sides share one space.

It connects directly to Postgres + Ollama (not via the HTTP API) because it is an
offline maintenance operation. It is idempotent: re-running simply recomputes the
same vectors.

Usage:
    uv run python scripts/reembed_corpus.py [--batch-size 64] [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx
from sqlalchemy import text

from superbrain.app.infrastructure.db.engine import (
    dispose_engine,
    get_session_factory,
    init_engine,
)
from superbrain.app.infrastructure.embeddings.ollama_embedder import OllamaEmbedder
from superbrain.settings import get_settings


def _vector_literal(embedding: list[float]) -> str:
    """Render a float vector as the pgvector text literal '[a,b,c]'."""
    return "[" + ",".join(str(float(v)) for v in embedding) + "]"


async def reembed(batch_size: int, limit: int | None, dry_run: bool) -> None:
    settings = get_settings()
    init_engine(settings.database_url)
    session_factory = get_session_factory()

    async with httpx.AsyncClient() as http_client:
        embedder = OllamaEmbedder(settings=settings, http_client=http_client)

        async with session_factory() as session:
            select_sql = "SELECT id, content FROM chunks ORDER BY id"
            if limit is not None:
                select_sql += f" LIMIT {int(limit)}"
            rows = (await session.execute(text(select_sql))).fetchall()

        total = len(rows)
        print(
            f"re-embedding {total} chunks with model={settings.ollama_embedding_model} "
            f"(prefix=search_document:, batch_size={batch_size}, dry_run={dry_run})"
        )
        if total == 0:
            print("no chunks found — nothing to do.")
            return

        started = time.monotonic()
        done = 0
        for start in range(0, total, batch_size):
            batch = rows[start : start + batch_size]
            contents = [r.content for r in batch]
            # input_type="document" applies the search_document: prefix inside the embedder.
            embeddings = await embedder.embed(contents, input_type="document")

            if not dry_run:
                async with session_factory() as session:
                    for row, emb in zip(batch, embeddings):
                        await session.execute(
                            text(
                                "UPDATE chunks SET embedding = CAST(:vec AS vector) "
                                "WHERE id = :id"
                            ),
                            {"vec": _vector_literal(emb), "id": row.id},
                        )
                    await session.commit()

            done += len(batch)
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed else 0.0
            print(f"  {done}/{total} chunks  ({rate:.1f}/s)")

    await dispose_engine()
    verb = "would re-embed" if dry_run else "re-embedded"
    print(f"done — {verb} {total} chunks in {time.monotonic() - started:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-embed all chunks with the nomic document prefix."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="chunks per embed call"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap chunks processed (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="embed but skip the UPDATE"
    )
    args = parser.parse_args()
    asyncio.run(reembed(args.batch_size, args.limit, args.dry_run))


if __name__ == "__main__":
    main()

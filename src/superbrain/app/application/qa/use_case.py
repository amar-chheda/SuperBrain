"""QA use case: decompose -> multi-probe recall -> SLM rerank gate -> grounded answer."""

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import structlog

from superbrain.app.application.metrics import MetricsRecorder
from superbrain.app.application.ports import LLMPort
from superbrain.app.application.qa.answer_generator import generate_answer
from superbrain.app.application.qa.evidence_builder import (
    Evidence,
    build_evidence_set,
    check_evidence_sufficiency,
)
from superbrain.app.application.qa.query_analysis import (
    QueryAnalysis,
    analyze_query,
    raw_analysis,
)
from superbrain.app.application.qa.reranker import rerank
from superbrain.app.application.retrieval.fusion import reciprocal_rank_fusion
from superbrain.app.application.retrieval.vector_retriever import VectorRetriever
from superbrain.app.domain.entities import QueryLog
from superbrain.app.domain.repositories import ArticleRepository, QueryLogRepository
from superbrain.app.infrastructure.crawlers.url_utils import canonicalise_url
from superbrain.app.infrastructure.db.repositories.chunk_retrieval_repo import (
    ChunkRetrievalRepository,
    RankedChunk,
)
from superbrain.settings import Settings

log = structlog.get_logger(__name__)


@dataclass
class Citation:
    number: int
    chunk_id: UUID
    article_title: str | None
    article_url: str
    excerpt: str


@dataclass
class QAResult:
    answer: str | None
    citations: list[Citation]
    aborted: bool
    abort_reason: str | None = None
    abort_kind: str | None = None  # e.g. "url_not_ingested"; lets callers tailor the reply
    retrieval_latency_ms: int = 0
    answer_latency_ms: int = 0


class AskQuestionUseCase:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        llm: LLMPort,
        query_log_repo: QueryLogRepository,
        metrics: MetricsRecorder,
        settings: Settings,
        article_repo: ArticleRepository,
        chunk_repo: ChunkRetrievalRepository,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._llm = llm
        self._query_log_repo = query_log_repo
        self._metrics = metrics
        self._settings = settings
        self._article_repo = article_repo
        self._chunk_repo = chunk_repo

    async def execute(self, question: str) -> QAResult:
        # Stage 1 — decompose the ask into routed parts (timed separately).
        t_analyze = time.monotonic()
        analysis = await self._analyze(question)
        analysis_ms = int((time.monotonic() - t_analyze) * 1000)
        if analysis.fell_back:
            self._metrics.increment("qa_query_analysis_fallback_total")

        # Intent routing: a question about a specific article goes to a direct
        # canonical-URL lookup; everything else is topic search.
        if analysis.intent == "summarize_url" and analysis.url:
            return await self._summarize_url(question, analysis, analysis_ms)
        return await self._topic_search(question, analysis, analysis_ms)

    async def _analyze(self, question: str) -> QueryAnalysis:
        """Decompose the question with the thinking model (or deterministic split)."""
        if not self._settings.qa_query_analysis_enabled:
            return raw_analysis(question)
        return await analyze_query(
            self._llm,
            model=self._settings.ollama_query_analysis_model,
            question=question,
        )

    async def _topic_search(
        self, question: str, analysis: QueryAnalysis, analysis_ms: int
    ) -> QAResult:
        """Multi-probe recall -> SLM rerank precision gate -> grounded answer."""
        s = self._settings

        # Stage 2 — recall. Probe on the CLEAN search_query (never the raw ask);
        # HyDE passage only widens the pool; BM25 runs OR-semantics on keywords.
        probes = [analysis.search_query]
        passage = (analysis.hypothetical_passage or "").strip()
        if passage and passage != analysis.search_query.strip():
            probes.append(passage)

        t_ret = time.monotonic()
        vector_lists = await self._vector_retriever.retrieve_multi(
            probes, top_k=s.qa_retrieval_top_k
        )
        bm25_chunks = await self._chunk_repo.find_by_text(
            analysis.keywords, top_k=s.qa_retrieval_top_k
        )
        retrieval_ms = int((time.monotonic() - t_ret) * 1000)
        self._metrics.observe("retrieval_latency_ms", retrieval_ms)

        best_vector = max((c.similarity_score for lst in vector_lists for c in lst), default=0.0)
        best_bm25 = max((c.similarity_score for c in bm25_chunks), default=0.0)

        # Cheap recall floor: if both legs are empty/weak, refuse before paying for
        # a rerank call — the topic simply isn't in the knowledge base.
        if best_vector < s.qa_min_vector_similarity and best_bm25 < s.qa_min_bm25_score:
            reason = (
                f"No credible match — best vector {best_vector:.3f} < {s.qa_min_vector_similarity} "
                f"and best lexical {best_bm25:.3f} < {s.qa_min_bm25_score}; topic likely absent"
            )
            self._metrics.increment("qa_aborted_low_relevance_total")
            return await self._abort(
                question, reason,
                stage_latencies={"analysis": analysis_ms, "retrieval": retrieval_ms, "rerank": 0},
                gate={"decision": "refuse", "stage": "recall_floor", "reason": reason,
                      "best_vector": round(best_vector, 4), "best_bm25": round(best_bm25, 4)},
                trace_parts=(analysis, question, vector_lists, bm25_chunks, [], [], {}, False),
            )

        # Stage 3 — fuse all legs into a candidate pool for reranking.
        fused = reciprocal_rank_fusion(*vector_lists, bm25_chunks, top_n=s.qa_rerank_pool_size)

        # Stage 4 — precision gate. Rerank the pool against the CLEAN query.
        rerank_ms = 0
        rerank_by_id: dict[UUID, float] = {}
        rerank_fell_back = False
        ranked = list(fused)  # default order if rerank disabled/failed
        top_rerank: float | None = None

        if s.qa_rerank_enabled and fused:
            t_rr = time.monotonic()
            result = await rerank(
                self._llm, model=s.ollama_rerank_model,
                query=analysis.search_query, chunks=fused,
            )
            rerank_ms = int((time.monotonic() - t_rr) * 1000)
            self._metrics.observe("rerank_latency_ms", rerank_ms)
            rerank_fell_back = result.fell_back
            if result.fell_back:
                self._metrics.increment("qa_rerank_fallback_total")
            else:
                pairs = sorted(zip(fused, result.scores), key=lambda x: x[1], reverse=True)
                rerank_by_id = {c.id: sc for c, sc in pairs}
                ranked = [c for c, _ in pairs]
                top_rerank = pairs[0][1] if pairs else 0.0

        latencies = {"analysis": analysis_ms, "retrieval": retrieval_ms, "rerank": rerank_ms}

        # The rerank gate: refuse when nothing is actually relevant enough. This is
        # the precision check the cosine/RRF floor cannot make.
        if top_rerank is not None and top_rerank < s.qa_min_rerank_score:
            reason = (
                f"No sufficiently relevant evidence — best relevance {top_rerank:.2f} "
                f"< {s.qa_min_rerank_score} (reranked {len(fused)} candidates)"
            )
            self._metrics.increment("qa_aborted_low_rerank_total")
            return await self._abort(
                question, reason,
                stage_latencies=latencies,
                gate={"decision": "refuse", "stage": "rerank", "reason": reason,
                      "best_vector": round(best_vector, 4), "best_bm25": round(best_bm25, 4),
                      "top_rerank": round(top_rerank, 4), "rerank_fell_back": rerank_fell_back},
                trace_parts=(analysis, question, vector_lists, bm25_chunks, fused, [],
                             rerank_by_id, rerank_fell_back),
            )

        # Evidence selection: keep only high-relevance chunks (no scattered grab-bag).
        if rerank_by_id:
            kept = [c for c in ranked if rerank_by_id.get(c.id, 0.0) >= s.qa_rerank_keep_score]
            evidence_chunks = (kept or ranked[:1])[: s.qa_evidence_top_n]
        else:
            evidence_chunks = ranked[: s.qa_evidence_top_n]  # rerank disabled/failed → RRF order

        evidence = build_evidence_set(evidence_chunks)
        is_sufficient, reason = check_evidence_sufficiency(evidence)
        if not is_sufficient:
            return await self._abort(
                question, reason,
                stage_latencies=latencies,
                gate={"decision": "refuse", "stage": "evidence_sufficiency", "reason": reason,
                      "top_rerank": round(top_rerank, 4) if top_rerank is not None else None,
                      "rerank_fell_back": rerank_fell_back},
                trace_parts=(analysis, question, vector_lists, bm25_chunks, fused,
                             evidence, rerank_by_id, rerank_fell_back),
            )

        gate = {"decision": "answer", "stage": "answered",
                "best_vector": round(best_vector, 4), "best_bm25": round(best_bm25, 4),
                "top_rerank": round(top_rerank, 4) if top_rerank is not None else None,
                "rerank_fell_back": rerank_fell_back}
        return await self._answer(
            question, analysis, evidence, latencies, gate,
            trace_parts=(analysis, question, vector_lists, bm25_chunks, fused,
                         evidence, rerank_by_id, rerank_fell_back),
        )

    async def _summarize_url(
        self, question: str, analysis: QueryAnalysis, analysis_ms: int
    ) -> QAResult:
        """Answer about a specific article via direct canonical-URL lookup.

        Grounds the answer in that one article's chunks (no rerank — it's an exact
        match). If the article is not ingested, refuse and tell the user to add it.
        """
        url = analysis.url or ""
        t_ret = time.monotonic()
        article = await self._article_repo.find_by_canonical_url(canonicalise_url(url))
        if article is None:
            article = await self._article_repo.find_by_canonical_url(url)
        chunks = (
            await self._chunk_repo.find_by_article(
                article.id, limit=self._settings.qa_url_max_chunks
            )
            if article is not None and article.status == "succeeded"
            else []
        )
        retrieval_ms = int((time.monotonic() - t_ret) * 1000)
        latencies = {"analysis": analysis_ms, "retrieval": retrieval_ms, "rerank": 0}

        if not chunks:
            reason = (
                f"I haven't ingested that article yet ({url}). Send me the URL on "
                "its own to add it, then ask again."
            )
            self._metrics.increment("qa_aborted_url_not_ingested_total")
            return await self._abort(
                question, reason,
                stage_latencies=latencies,
                gate={"decision": "refuse", "stage": "url_lookup", "reason": reason, "url": url},
                trace_parts=(analysis, question, [], [], [], [], {}, False),
                abort_kind="url_not_ingested",
            )

        evidence = build_evidence_set(chunks)
        gate = {"decision": "answer", "stage": "url_lookup", "url": url,
                "article_id": str(article.id)}
        return await self._answer(
            question, analysis, evidence, latencies, gate,
            trace_parts=(analysis, question, [], [], chunks, evidence, {}, False),
        )

    async def _abort(
        self,
        question: str,
        reason: str,
        *,
        stage_latencies: dict[str, int],
        gate: dict[str, Any],
        trace_parts: tuple,
        abort_kind: str | None = None,
    ) -> QAResult:
        """Log a refusal (with full trace) and return an aborted result."""
        retrieval_ms = stage_latencies.get("retrieval", 0)
        log.info("qa.aborted", reason=reason, kind=abort_kind, stage=gate.get("stage"),
                 question=question[:100])
        self._metrics.increment("qa_aborted_total")
        trace = _build_trace(*trace_parts, gate=gate, stage_latencies=stage_latencies)
        await self._query_log_repo.save(
            QueryLog(
                id=uuid4(),
                question=question,
                answer=None,
                evidence_chunk_ids=[],
                retrieval_latency_ms=retrieval_ms,
                answer_latency_ms=0,
                aborted=True,
                abort_reason=reason,
                created_at=datetime.now(tz=UTC),
                retrieval_trace=trace,
            )
        )
        return QAResult(
            answer=None, citations=[], aborted=True, abort_reason=reason,
            abort_kind=abort_kind, retrieval_latency_ms=retrieval_ms,
        )

    async def _answer(
        self,
        question: str,
        analysis: QueryAnalysis,
        evidence: list[Evidence],
        stage_latencies: dict[str, int],
        gate: dict[str, Any],
        *,
        trace_parts: tuple,
    ) -> QAResult:
        """Generate a grounded answer (shaped by directives), log it, and return."""
        t_answer = time.monotonic()
        answer_text, cited_pairs, prompt_sent = await generate_answer(
            self._llm,
            model=self._settings.ollama_qa_model,
            question=question,
            evidence=evidence,
            answer_directives=analysis.answer_directives,
        )
        answer_ms = int((time.monotonic() - t_answer) * 1000)
        self._metrics.observe("answer_latency_ms", answer_ms)
        self._metrics.increment("qa_success_total")
        stage_latencies = {**stage_latencies, "answer": answer_ms}

        evidence_by_id = {e.chunk_id: e for e in evidence}
        citations = [
            Citation(
                number=n,
                chunk_id=chunk_id,
                article_title=evidence_by_id[chunk_id].article_title,
                article_url=evidence_by_id[chunk_id].article_url,
                excerpt=(
                    evidence_by_id[chunk_id].content[:200] + "..."
                    if len(evidence_by_id[chunk_id].content) > 200
                    else evidence_by_id[chunk_id].content
                ),
            )
            for n, chunk_id in cited_pairs
            if chunk_id in evidence_by_id
        ]

        retrieval_ms = stage_latencies.get("retrieval", 0)
        trace = _build_trace(*trace_parts, gate=gate, stage_latencies=stage_latencies,
                             prompt_sent=prompt_sent)
        await self._query_log_repo.save(
            QueryLog(
                id=uuid4(),
                question=question,
                answer=answer_text,
                evidence_chunk_ids=[chunk_id for _, chunk_id in cited_pairs],
                retrieval_latency_ms=retrieval_ms,
                answer_latency_ms=answer_ms,
                aborted=False,
                abort_reason=None,
                created_at=datetime.now(tz=UTC),
                retrieval_trace=trace,
            )
        )
        log.info("qa.answered", question=question[:100], citation_count=len(citations),
                 stage_latencies=stage_latencies)
        return QAResult(
            answer=answer_text, citations=citations, aborted=False,
            retrieval_latency_ms=retrieval_ms, answer_latency_ms=answer_ms,
        )


def _hit(c: Any, rerank_by_id: dict[UUID, float]) -> dict[str, Any]:
    return {
        "chunk_id": str(c.id),
        "score": round(c.similarity_score, 4),
        "rerank_score": rerank_by_id.get(c.id),
        "article": c.title,
        "url": c.url,
        "content": (c.content or "")[:300],
    }


def _build_trace(
    analysis: QueryAnalysis | None,
    raw_question: str,
    vector_lists: list[list[RankedChunk]],
    bm25_chunks: list[RankedChunk],
    fused: list[RankedChunk],
    evidence: list[Evidence],
    rerank_by_id: dict[UUID, float],
    rerank_fell_back: bool,
    *,
    gate: dict[str, Any] | None = None,
    stage_latencies: dict[str, int] | None = None,
    prompt_sent: str | None = None,
) -> dict[str, Any]:
    """Serialise the full pipeline (decompose -> recall -> rerank -> gate) for query_logs.

    vector_lists[0] is the clean search_query probe; [1] (if present) is the HyDE probe.
    """
    raw_hits = vector_lists[0] if len(vector_lists) > 0 else []
    hyde_hits = vector_lists[1] if len(vector_lists) > 1 else []
    return {
        "query_analysis": (
            {
                "raw_question": raw_question,
                "search_query": analysis.search_query,
                "keywords": analysis.keywords,
                "hypothetical_passage": analysis.hypothetical_passage,
                "answer_directives": analysis.answer_directives,
                "intent": analysis.intent,
                "url": analysis.url,
                "fell_back": analysis.fell_back,
            }
            if analysis is not None
            else None
        ),
        "stage_latencies_ms": stage_latencies or {},
        "gate": gate or {},
        "rerank_fell_back": rerank_fell_back,
        "vector_hits": [_hit(c, rerank_by_id) for c in raw_hits],
        "vector_hyde_hits": [_hit(c, rerank_by_id) for c in hyde_hits],
        "bm25_hits": [_hit(c, rerank_by_id) for c in bm25_chunks],
        "fused": [
            {
                "chunk_id": str(c.id),
                "rrf_score": round(c.rrf_score, 6),
                "rerank_score": rerank_by_id.get(c.id),
                "article": c.title,
                "url": c.url,
            }
            for c in fused
        ],
        "evidence": [
            {
                "chunk_id": str(e.chunk_id),
                "rrf_score": round(e.rrf_score, 6),
                "rerank_score": rerank_by_id.get(e.chunk_id),
                "article": e.article_title,
                "url": e.article_url,
                "content": (e.content or "")[:500],
            }
            for e in evidence
        ],
        "prompt_sent": prompt_sent,
    }

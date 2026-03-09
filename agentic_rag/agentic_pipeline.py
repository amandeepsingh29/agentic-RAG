from __future__ import annotations

from dataclasses import dataclass, replace
import re

from .pipeline import RAGPipeline
from .models import AnswerResult


@dataclass
class AgenticRAGPipeline(RAGPipeline):
    """Bounded agentic controller over the classic pgvector RAG pipeline."""

    max_iterations: int = 3

    def ask(self, question: str, filters: dict[str, object] | None = None) -> AnswerResult:
        pending = [question.strip()]
        seen_queries: set[str] = set()
        best_hits: list = []
        trace: list[dict] = []

        for iteration in range(self.max_iterations):
            if not pending:
                break
            query = pending.pop(0)
            if query.lower() in seen_queries:
                continue
            seen_queries.add(query.lower())

            query_embedding = self.embedding_model.encode([query])[0]
            with self._store_lock:
                hits = self.store.query(query_embedding, top_k=self.top_k, filters=filters)
            trace.append(
                {
                    "step": iteration + 1,
                    "action": "retrieve",
                    "query": query,
                    "hits": len(hits),
                    "top_score": round(hits[0].score, 4) if hits else 0.0,
                }
            )

            if hits and (not best_hits or hits[0].score > best_hits[0].score):
                best_hits = hits
            sufficient = bool(hits) and not self._should_abstain(question, hits)
            trace.append(
                {
                    "step": iteration + 1,
                    "action": "evaluate_evidence",
                    "sufficient": sufficient,
                }
            )
            if sufficient:
                result = self._answer_from_hits(question, hits)
                return replace(result, trace=trace)

            rewrites = self._rewrite_query(question, hits, seen_queries)
            if rewrites:
                trace.append({"step": iteration + 1, "action": "rewrite_query", "queries": rewrites})
                pending.extend(rewrites)

        trace.append({"step": len(trace) + 1, "action": "refuse", "reason": "Evidence remained insufficient."})
        return AnswerResult(
            question=question,
            answer="I don't know.",
            sources=[],
            confidence=float(best_hits[0].score) if best_hits else 0.0,
            abstained=True,
            refusal_reason="The agent tried bounded retrieval steps but could not find sufficient evidence.",
            trace=trace,
        )

    @staticmethod
    def _rewrite_query(question: str, hits: list, seen_queries: set[str]) -> list[str]:
        titles: list[str] = []
        for hit in hits[:3]:
            title = hit.chunk.title.strip()
            if title and title.lower() not in {item.lower() for item in titles}:
                titles.append(title)

        ambiguous = bool(re.search(r"\b(it|they|them|this|that|these|those)\b", question.lower()))
        if not ambiguous and hits and hits[0].score >= 0.08:
            return []

        rewrites = [f"{question} in Kubernetes {title}" for title in titles[:2]]
        return [query for query in rewrites if query.lower() not in seen_queries]

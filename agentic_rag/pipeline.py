from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from threading import RLock

from .embeddings import EmbeddingModel
from .ingestion import chunk_documents, load_documents
from .llm import OpenRouterLLM
from .models import AnswerResult, CorpusStats
from .vector_store import PersistentVectorStore


@dataclass
class RAGPipeline:
    persist_dir: str | Path
    embedding_model: EmbeddingModel | None = None
    llm: OpenRouterLLM | None = None
    top_k: int = 5
    min_score: float = 0.12
    _store_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.embedding_model = self.embedding_model or EmbeddingModel()
        self.store = PersistentVectorStore(self.persist_dir)

    def ingest(self, documents_path: str | Path) -> CorpusStats:
        documents = load_documents([documents_path])
        chunks = chunk_documents(documents)
        if chunks:
            embeddings = self.embedding_model.encode([chunk.text for chunk in chunks])
            # Compute embeddings before replacing the current index so a failed
            # model download or encoding run does not destroy a usable corpus.
            with self._store_lock:
                self.store.reset()
                self.store.add(chunks, embeddings)
        else:
            with self._store_lock:
                self.store.reset()
        sources = sorted({document.source for document in documents})
        return CorpusStats(documents=len(documents), chunks=len(chunks), sources=sources)

    def ask(self, question: str, filters: dict[str, object] | None = None) -> AnswerResult:
        query_embedding = self.embedding_model.encode([question])[0]
        with self._store_lock:
            hits = self.store.query(query_embedding, top_k=self.top_k, filters=filters)
        if not hits:
            return AnswerResult(
                question=question,
                answer="I don't know.",
                sources=[],
                confidence=0.0,
                abstained=True,
                refusal_reason="No retrieved chunk was strong enough to answer safely.",
            )

        if self._should_abstain(question, hits):
            return AnswerResult(
                question=question,
                answer="I don't know.",
                sources=[],
                confidence=float(hits[0].score),
                abstained=True,
                refusal_reason="Retrieved evidence did not sufficiently overlap with the question.",
            )

        context_parts: list[str] = []
        sources: list[dict] = []
        for hit in hits:
            snippet = f"[{Path(hit.chunk.source).name}] {hit.chunk.text}"
            context_parts.append(snippet)
            sources.append(
                {
                    "source": Path(hit.chunk.source).name,
                    "title": hit.chunk.title,
                    "chunk_id": hit.chunk.chunk_id,
                    "score": round(hit.score, 4),
                    "position": hit.chunk.position,
                }
            )

        context = "\n\n".join(context_parts)
        if self.llm is None:
            self.llm = OpenRouterLLM()
        answer = self.llm.generate(question, context)
        confidence = max(0.0, min(1.0, hits[0].score))

        if answer.strip().lower() in {"i don't know.", "i don't know", "i do not know."}:
            return AnswerResult(
                question=question,
                answer=answer,
                sources=[],
                confidence=confidence,
                abstained=True,
                refusal_reason="The language model declined because the context was insufficient.",
            )

        return AnswerResult(
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            abstained=False,
        )

    def _should_abstain(self, question: str, hits: list) -> bool:
        if hits[0].score < self.min_score:
            return True

        question_terms = self._content_terms(question)
        if not question_terms:
            return True

        context_terms = set()
        for hit in hits[:3]:
            context_terms.update(self._content_terms(hit.chunk.text))
            context_terms.update(self._content_terms(hit.chunk.title))

        overlap = question_terms & context_terms
        return len(overlap) < 2

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        stopwords = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "how",
            "why",
            "is",
            "are",
            "was",
            "were",
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "for",
            "in",
            "on",
            "with",
            "do",
            "does",
            "did",
            "it",
            "this",
            "that",
            "used",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
            if len(token) > 3 and token not in stopwords and not token.endswith("ing")
        }

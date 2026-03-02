from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import chunk_document
from .models import Chunk, Document, RetrievalHit


@dataclass
class HybridRetriever:
    word_vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    )
    char_vectorizer: TfidfVectorizer = field(
        default_factory=lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    )
    chunks: list[Chunk] = field(default_factory=list)
    _word_matrix: Any = None
    _char_matrix: Any = None

    def index_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(chunk_document(document))

        self.chunks = chunks
        corpus = [chunk.text for chunk in chunks] or [""]
        self._word_matrix = self.word_vectorizer.fit_transform(corpus)
        self._char_matrix = self.char_vectorizer.fit_transform(corpus)
        return chunks

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievalHit]:
        if not self.chunks:
            return []

        word_query = self.word_vectorizer.transform([question])
        char_query = self.char_vectorizer.transform([question])
        word_scores = (self._word_matrix @ word_query.T).toarray().ravel()
        char_scores = (self._char_matrix @ char_query.T).toarray().ravel()

        combined = 0.65 * normalize_scores(word_scores) + 0.35 * normalize_scores(char_scores)
        order = np.argsort(-combined)[:top_k]

        hits: list[RetrievalHit] = []
        for rank, index in enumerate(order, start=1):
            chunk = self.chunks[index]
            score = float(combined[index])
            rationale = "hybrid lexical + character n-gram relevance"
            hits.append(RetrievalHit(chunk=chunk, score=score, rank=rank, rationale=rationale))
        return hits


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(float)
    max_score = float(scores.max()) if scores.size else 0.0
    if max_score <= 0:
        return scores
    return scores / max_score

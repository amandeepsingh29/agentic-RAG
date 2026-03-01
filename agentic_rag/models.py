from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    rank: int
    rationale: str = ""


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    confidence: float
    abstained: bool
    refusal_reason: str | None = None


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: ChunkRecord
    score: float


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    abstained: bool
    refusal_reason: str | None = None


@dataclass(frozen=True)
class CorpusStats:
    documents: int
    chunks: int
    sources: list[str]


def to_path_str(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


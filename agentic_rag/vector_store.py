from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from .models import ChunkRecord, RetrievedChunk


@dataclass
class PersistentVectorStore:
    """Qdrant-backed vector store using Qdrant's persistent local mode."""

    persist_dir: str | Path
    collection_name: str = "knowledge_base"

    def __post_init__(self) -> None:
        self.persist_dir = Path(self.persist_dir).expanduser().resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.persist_dir))

    def reset(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def add(self, chunks: list[ChunkRecord], embeddings: np.ndarray) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("Every chunk must have exactly one embedding.")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
        )
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=vector.tolist(),
                payload=self._chunk_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not self.client.collection_exists(self.collection_name):
            return []

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=np.asarray(query_embedding, dtype=np.float32).tolist(),
            query_filter=self._build_filter(filters),
            limit=top_k,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                chunk=self._chunk_from_payload(point.payload or {}),
                score=float(point.score),
            )
            for point in response.points
        ]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)

    def export_manifest(self, path: str | Path) -> None:
        payload = {
            "collection": self.collection_name,
            "count": self.count(),
            "storage": str(self.persist_dir),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _chunk_to_payload(chunk: ChunkRecord) -> dict[str, Any]:
        payload = asdict(chunk)
        # Keep commonly filtered fields at the top level for efficient Qdrant payload indexes.
        payload.update(chunk.metadata)
        return payload

    @staticmethod
    def _chunk_from_payload(payload: dict[str, Any]) -> ChunkRecord:
        metadata_keys = {"chunk_id", "doc_id", "source", "title", "text", "position"}
        metadata = dict(payload.get("metadata") or {})
        metadata.update({key: value for key, value in payload.items() if key not in metadata_keys and key != "metadata"})
        return ChunkRecord(
            chunk_id=str(payload["chunk_id"]),
            doc_id=str(payload["doc_id"]),
            source=str(payload["source"]),
            title=str(payload["title"]),
            text=str(payload["text"]),
            position=int(payload["position"]),
            metadata=metadata,
        )

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
        if not filters:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in filters.items()
            ]
        )

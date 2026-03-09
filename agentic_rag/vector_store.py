from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import re
import uuid

import numpy as np
import psycopg
from psycopg import sql

from .models import ChunkRecord, RetrievedChunk


@dataclass
class PersistentVectorStore:
    """PostgreSQL + pgvector store for vectors, chunks, and metadata."""

    persist_dir: str | Path
    collection_name: str = "knowledge_base"
    dsn: str | None = None
    embedding_dim: int = 384

    def __post_init__(self) -> None:
        self.persist_dir = Path(self.persist_dir).expanduser().resolve()
        self.dsn = self.dsn or os.getenv("DATABASE_URL", "postgresql://127.0.0.1:5432/rag_assistant")
        self.table_name = self._safe_identifier(self.collection_name)
        self.client = psycopg.connect(self.dsn, autocommit=True)
        self._ensure_schema()

    def reset(self) -> None:
        self.client.execute(sql.SQL("DELETE FROM {}" ).format(sql.Identifier(self.table_name)))

    def add(self, chunks: list[ChunkRecord], embeddings: np.ndarray) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("Every chunk must have exactly one embedding.")
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim}-dimensional embeddings, got {vectors.shape[1]}.")

        rows = [
            (
                str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                chunk.chunk_id,
                chunk.doc_id,
                chunk.source,
                chunk.title,
                chunk.text,
                chunk.position,
                json.dumps(chunk.metadata),
                self._vector_literal(vector),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        statement = sql.SQL(
            "INSERT INTO {} "
            "(id, chunk_id, doc_id, source, title, content, position, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector) "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            "doc_id = EXCLUDED.doc_id, source = EXCLUDED.source, title = EXCLUDED.title, "
            "content = EXCLUDED.content, position = EXCLUDED.position, metadata = EXCLUDED.metadata, "
            "embedding = EXCLUDED.embedding"
        ).format(sql.Identifier(self.table_name))
        with self.client.cursor() as cursor:
            cursor.executemany(statement, rows)

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        vector = self._vector_literal(query_embedding)
        where_clauses: list[sql.Composed] = []
        filter_params: list[Any] = []
        column_filters = {"chunk_id", "doc_id", "source", "title", "position"}

        for key, value in (filters or {}).items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid metadata filter: {key}")
            if key in column_filters:
                where_clauses.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
                filter_params.append(value)
            else:
                where_clauses.append(sql.SQL("metadata ->> %s = %s"))
                filter_params.extend([key, str(value)])

        where = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses) if where_clauses else sql.SQL("")
        statement = sql.SQL(
            "SELECT chunk_id, doc_id, source, title, content, position, metadata, "
            "1 - (embedding <=> %s::vector) AS score "
            "FROM {}{} ORDER BY embedding <=> %s::vector LIMIT %s"
        ).format(sql.Identifier(self.table_name), where)
        params = [vector, *filter_params, vector, top_k]

        with self.client.cursor() as cursor:
            cursor.execute(statement, params)
            rows = cursor.fetchall()

        return [
            RetrievedChunk(
                chunk=ChunkRecord(
                    chunk_id=row[0],
                    doc_id=row[1],
                    source=row[2],
                    title=row[3],
                    text=row[4],
                    position=row[5],
                    metadata=row[6] or {},
                ),
                score=float(row[7]),
            )
            for row in rows
        ]

    def count(self) -> int:
        statement = sql.SQL("SELECT count(*) FROM {}" ).format(sql.Identifier(self.table_name))
        return int(self.client.execute(statement).fetchone()[0])

    def export_manifest(self, path: str | Path) -> None:
        payload = {
            "table": self.table_name,
            "count": self.count(),
            "database": self.dsn.rsplit("/", 1)[-1],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def close(self) -> None:
        self.client.close()

    def _ensure_schema(self) -> None:
        self.client.execute("CREATE EXTENSION IF NOT EXISTS vector")
        table = sql.Identifier(self.table_name)
        self.client.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} ("
                "id uuid PRIMARY KEY, "
                "chunk_id text NOT NULL UNIQUE, "
                "doc_id text NOT NULL, "
                "source text NOT NULL, "
                "title text NOT NULL, "
                "content text NOT NULL, "
                "position integer NOT NULL, "
                "metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb, "
                "embedding vector({}) NOT NULL"
                ")"
            ).format(table, sql.SQL(str(self.embedding_dim)))
        )
        self.client.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops)").format(
                sql.Identifier(f"{self.table_name}_embedding_hnsw"), table
            )
        )
        self.client.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING gin (metadata)").format(
                sql.Identifier(f"{self.table_name}_metadata_gin"), table
            )
        )

    @staticmethod
    def _safe_identifier(value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(f"Invalid PostgreSQL table name: {value}")
        return value

    @staticmethod
    def _vector_literal(vector: np.ndarray) -> str:
        return "[" + ",".join(str(float(value)) for value in vector) + "]"

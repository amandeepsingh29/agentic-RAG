from __future__ import annotations

from .models import ChunkRecord, DocumentRecord


def chunk_document(
    document: DocumentRecord,
    chunk_size: int = 650,
    overlap: int = 100,
) -> list[ChunkRecord]:
    text = document.text.strip()
    if not text:
        return []

    paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[ChunkRecord] = []
    buffer = ""
    position = 0
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer:
            chunks.append(_build_chunk(document, buffer, position))
            position += 1

        if len(paragraph) <= chunk_size:
            buffer = paragraph
        else:
            start = 0
            while start < len(paragraph):
                window = paragraph[start : start + chunk_size]
                chunks.append(_build_chunk(document, window, position))
                position += 1
                start += max(1, chunk_size - overlap)
            buffer = ""

    if buffer:
        chunks.append(_build_chunk(document, buffer, position))

    return chunks


def _build_chunk(document: DocumentRecord, text: str, position: int) -> ChunkRecord:
    chunk_id = f"{document.doc_id}:{position}"
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=document.doc_id,
        source=document.source,
        title=document.title,
        text=text.strip(),
        position=position,
        metadata=dict(document.metadata),
    )

from __future__ import annotations

from .models import Chunk, Document


def chunk_document(document: Document, max_words: int = 140, overlap_words: int = 30) -> list[Chunk]:
    words = document.text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(1, max_words - overlap_words)
    position = 0
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if not window:
            break
        text = " ".join(window).strip()
        chunk_id = f"{document.doc_id}:{position}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                source=document.source,
                text=text,
                position=position,
                metadata=dict(document.metadata),
            )
        )
        position += 1
    return chunks


from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .models import Document

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    documents: list[Document] = []
    for path_like in paths:
        path = Path(path_like)
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    documents.extend(load_documents([file_path]))
            continue

        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue

        documents.append(
            Document(
                doc_id=str(path.resolve()),
                source=str(path),
                text=text,
                metadata={"extension": suffix.lstrip(".")},
            )
        )
    return documents


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


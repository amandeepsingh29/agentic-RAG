from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import requests
from pypdf import PdfReader
import re

from .chunking import chunk_document
from .models import ChunkRecord, DocumentRecord, to_path_str

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}

KUBERNETES_DOCS = [
    (
        "configmap",
        "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/configuration/configmap.md",
    ),
    (
        "secret",
        "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/configuration/secret.md",
    ),
    (
        "service",
        "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/services-networking/service.md",
    ),
    (
        "pod-lifecycle",
        "https://raw.githubusercontent.com/kubernetes/website/main/content/en/docs/concepts/workloads/pods/pod-lifecycle.md",
    ),
]

DEFAULT_DOCUMENTS_DIR = Path("data/kubernetes")


def download_kubernetes_docs(destination: str | Path) -> list[Path]:
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for slug, url in KUBERNETES_DOCS:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        target = dest / f"{slug}.md"
        target.write_text(response.text, encoding="utf-8")
        written.append(target)
    return written


def refresh_default_documents(destination: str | Path | None = None) -> list[Path]:
    target = Path(destination) if destination is not None else DEFAULT_DOCUMENTS_DIR
    return download_kubernetes_docs(target)


def load_documents(paths: Iterable[str | Path]) -> list[DocumentRecord]:
    documents: list[DocumentRecord] = []
    for path_like in paths:
        path = Path(path_like)
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    documents.extend(load_documents([file_path]))
            continue

        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            text = _clean_markdown(path.read_text(encoding="utf-8")) if suffix in {".md", ".markdown"} else path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue

        documents.append(
            DocumentRecord(
                doc_id=to_path_str(path),
                source=to_path_str(path),
                title=path.stem.replace("-", " ").title(),
                text=text,
                metadata={"extension": suffix.lstrip("."), "filename": path.name},
            )
        )
    return documents


def chunk_documents(documents: Iterable[DocumentRecord]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def dump_documents_manifest(documents: Iterable[DocumentRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(document) for document in documents]
    path.write_text(json_dumps(payload), encoding="utf-8")


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, indent=2, ensure_ascii=False)


def _clean_markdown(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.S)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.M)
    cleaned = re.sub(r"^\s*[-*+]\s+", "- ", cleaned, flags=re.M)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.M)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()

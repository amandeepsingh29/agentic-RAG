from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .answering import AnswerEngine
from .ingestion import load_documents
from .retrieval import HybridRetriever

app = FastAPI(title="Agentic RAG", version="0.1.0")
retriever = HybridRetriever()
engine = AnswerEngine()


class AskRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    paths: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict[str, int]:
    documents = load_documents(payload.paths)
    chunks = retriever.index_documents(documents)
    return {"documents": len(documents), "chunks": len(chunks)}


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    hits = retriever.retrieve(payload.question)
    result = engine.answer(payload.question, hits)
    return result.__dict__


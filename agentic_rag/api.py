from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .ingestion import refresh_default_documents
from .agentic_pipeline import AgenticRAGPipeline
from .pipeline import RAGPipeline

app = FastAPI(title="RAG Documentation Assistant", version="1.0.0")
_pipeline: AgenticRAGPipeline | None = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = (PROJECT_ROOT / "data").resolve()
WEBAPP_PATH = PROJECT_ROOT / "webapp/index.html"
SHOWCASE_PATH = PROJECT_ROOT / "webapp/showcase.html"


class IngestRequest(BaseModel):
    documents_path: str = Field(default="data/kubernetes", min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    filters: dict[str, str | int | bool] | None = None


class RefreshRequest(BaseModel):
    destination: str | None = None


def _data_path(value: str | None, default: Path) -> Path:
    candidate = (PROJECT_ROOT / value).resolve() if value else default
    if candidate != DATA_ROOT and DATA_ROOT not in candidate.parents:
        raise HTTPException(status_code=400, detail="Paths must stay inside the project data directory.")
    return candidate


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_model=None)
def home():
    if WEBAPP_PATH.exists():
        return FileResponse(WEBAPP_PATH)
    return HTMLResponse("<h1>RAG Documentation Assistant</h1><p>The web app is not available yet.</p>")


@app.get("/showcase", response_model=None)
def showcase():
    if SHOWCASE_PATH.exists():
        return FileResponse(SHOWCASE_PATH)
    return HTMLResponse("<h1>Showcase not available</h1>")


@app.post("/refresh-docs")
def refresh_docs(payload: RefreshRequest) -> dict:
    destination = _data_path(payload.destination, DATA_ROOT / "kubernetes")
    written = refresh_default_documents(destination)
    return {"downloaded": [str(path) for path in written]}


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict:
    documents_path = _data_path(payload.documents_path, DATA_ROOT / "kubernetes")
    stats = get_pipeline().ingest(documents_path)
    return stats.__dict__


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    try:
        result = get_pipeline().ask(payload.question, filters=payload.filters)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result.__dict__


@app.on_event("startup")
def warm_pipeline() -> None:
    try:
        get_pipeline()
    except HTTPException:
        pass


@app.on_event("shutdown")
def close_pipeline() -> None:
    if _pipeline is not None:
        _pipeline.store.close()


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = AgenticRAGPipeline(persist_dir=DATA_ROOT / "pgvector")
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _pipeline

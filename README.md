# Agentic RAG

An internal-knowledge RAG project built to answer questions from a small corpus of company files with citations, refusal behavior, and evaluation hooks.

## What it does

- Ingests text, markdown, and PDF documents
- Chunks documents with overlap and metadata
- Retrieves evidence with a hybrid ranking strategy
- Produces grounded answers with citations
- Refuses to answer when evidence is weak
- Exposes a small FastAPI service and a CLI

## Why this project exists

This repository is meant to demonstrate how to build a production-shaped RAG system that is safe enough for internal company knowledge search.

## Project layout

- `agentic_rag/` core package
- `samples/` example documents and questions
- `tests/` retrieval and policy tests

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn agentic_rag.api:app --reload
```


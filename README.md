# RAG Documentation Assistant

This project implements both a classic RAG baseline and a bounded agentic RAG pipeline:

Documents -> Chunking -> Embeddings -> Vector Store -> Retrieval -> LLM Generation

The agentic mode adds bounded query rewriting, iterative retrieval, evidence evaluation, and explicit refusal actions.

The knowledge base is built from real Kubernetes documentation downloaded from the official Kubernetes website repository.

## Features

- Downloads public documentation and stores it locally
- Chunks documents into retrieval-sized passages
- Creates embeddings with a sentence-transformer model
- Stores vectors in PostgreSQL with the pgvector extension
- Supports metadata-filtered retrieval
- Retrieves the most relevant chunks for a question
- Generates grounded answers through OpenRouter using `tencent/hy3:free`
- Refuses to answer when retrieval confidence is too low
- Agentic mode records retrieval, evidence evaluation, rewrite, and refusal steps

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENROUTER_API_KEY="your-key-here"
export OPENROUTER_HTTP_REFERER="http://localhost"
export OPENROUTER_TITLE="RAG Documentation Assistant"
rag-docs download-docs --output data/kubernetes
rag-docs ingest --documents data/kubernetes --index data/pgvector
rag-docs ask "How do Deployments manage Pods?" --mode agentic
rag-docs ask "How do Deployments manage Pods?" --mode classic
```

## API

```bash
uvicorn agentic_rag.api:app --host 127.0.0.1 --port 8080 --reload
```

## Web app

Open `http://127.0.0.1:8080/` for the user-facing documentation chat and `http://127.0.0.1:8080/showcase` for the project knowledge base.

PostgreSQL runs locally with the pgvector extension; Docker is not required. Set `DATABASE_URL` in `.env` to select the database.

The backend provides operational endpoints for:

- refreshing the real Kubernetes documentation corpus (`POST /refresh-docs`)
- building the retrieval index (`POST /ingest`)
- answering questions with citations (`POST /ask`)

The API uses agentic mode by default. The classic `RAGPipeline` remains available as a baseline for evaluation.

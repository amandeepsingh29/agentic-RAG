from pathlib import Path

import pytest

from agentic_rag.ingestion import download_kubernetes_docs
from agentic_rag.agentic_pipeline import AgenticRAGPipeline
from agentic_rag.evaluation import load_questions
from agentic_rag.pipeline import RAGPipeline


class FakeLLM:
    def generate(self, question: str, context: str) -> str:
        if "refund policy" in question.lower():
            return "I don't know."
        return "This answer is grounded in the retrieved Kubernetes documentation."


@pytest.fixture(scope="session")
def kubernetes_pipeline(tmp_path_factory: pytest.TempPathFactory) -> RAGPipeline:
    workspace = tmp_path_factory.mktemp("kubernetes_rag")
    docs_dir = workspace / "docs"
    index_dir = workspace / "index"
    download_kubernetes_docs(docs_dir)
    pipeline = RAGPipeline(persist_dir=index_dir, llm=FakeLLM())
    stats = pipeline.ingest(docs_dir)
    assert stats.documents >= 4
    assert stats.chunks >= 4
    return pipeline


def test_ingestion_and_retrieval(kubernetes_pipeline: RAGPipeline):
    result = kubernetes_pipeline.ask("How do ConfigMaps help with configuration?")
    assert result.abstained is False
    assert result.sources
    assert len(result.answer) > 10


def test_abstains_for_unrelated_question(kubernetes_pipeline: RAGPipeline):
    result = kubernetes_pipeline.ask("What is the refund policy for airline tickets?")
    assert result.abstained is True
    assert result.answer == "I don't know."


def test_handles_ambiguous_query(kubernetes_pipeline: RAGPipeline):
    result = kubernetes_pipeline.ask("How does it work?")
    assert isinstance(result.abstained, bool)
    assert result.answer


def test_agentic_pipeline_records_bounded_trace(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    index_dir = tmp_path / "qdrant"
    download_kubernetes_docs(docs_dir)
    pipeline = AgenticRAGPipeline(persist_dir=index_dir, llm=FakeLLM(), max_iterations=3)
    pipeline.ingest(docs_dir)

    result = pipeline.ask("What is the refund policy for airline tickets?")

    assert result.abstained is True
    assert result.trace
    assert result.trace[-1]["action"] == "refuse"
    assert len(result.trace) <= 7


def test_metadata_filter_retrieves_matching_documents(kubernetes_pipeline: RAGPipeline):
    result = kubernetes_pipeline.ask(
        "How do ConfigMaps help with configuration?",
        filters={"filename": "configmap.md"},
    )

    assert result.abstained is False
    assert result.sources
    assert {source["source"] for source in result.sources} == {"configmap.md"}


def test_evaluation_set_covers_three_way_comparison():
    questions = load_questions(Path("data/eval_queries.jsonl"))
    categories = {question.category for question in questions}

    assert len(questions) >= 12
    assert {"in-corpus", "agentic-recovery", "outside-corpus", "unsupported", "ambiguous"} <= categories

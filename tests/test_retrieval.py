from agentic_rag.answering import AnswerEngine
from agentic_rag.ingestion import load_documents
from agentic_rag.retrieval import HybridRetriever


def test_retrieval_returns_relevant_source():
    documents = load_documents(["samples/knowledge"])
    retriever = HybridRetriever()
    retriever.index_documents(documents)

    hits = retriever.retrieve("How do I request VPN access?")
    assert hits
    assert "it-onboarding.md" in hits[0].chunk.source


def test_abstains_on_missing_answer():
    documents = load_documents(["samples/knowledge"])
    retriever = HybridRetriever()
    retriever.index_documents(documents)

    result = AnswerEngine().answer("What is the travel reimbursement policy?", retriever.retrieve("What is the travel reimbursement policy?"))
    assert result.abstained is True


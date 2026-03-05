from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .answering import AnswerEngine
from .ingestion import load_documents
from .retrieval import HybridRetriever


@dataclass
class EvalRow:
    question: str
    expected_keyword: str
    abstained: bool
    top_source: str | None


def run_eval(corpus_path: str | Path, questions_path: str | Path) -> list[EvalRow]:
    documents = load_documents([corpus_path])
    retriever = HybridRetriever()
    retriever.index_documents(documents)
    engine = AnswerEngine()

    rows: list[EvalRow] = []
    with Path(questions_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            question = payload["question"]
            expected_keyword = payload.get("expected_keyword", "")
            result = engine.answer(question, retriever.retrieve(question))
            top_source = result.citations[0]["source"] if result.citations else None
            rows.append(
                EvalRow(
                    question=question,
                    expected_keyword=expected_keyword,
                    abstained=result.abstained,
                    top_source=top_source,
                )
            )
    return rows


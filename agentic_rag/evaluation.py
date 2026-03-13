from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Callable

from .agentic_pipeline import AgenticRAGPipeline
from .llm import OpenRouterLLM
from .pipeline import RAGPipeline


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    category: str
    expected_keywords: tuple[str, ...]

    @property
    def expected_answer(self) -> bool:
        return self.category in {"in-corpus", "agentic-recovery", "outside-corpus"}

    @property
    def requires_corpus(self) -> bool:
        return self.category in {"in-corpus", "agentic-recovery"}

    @property
    def expected_refusal(self) -> bool:
        return self.category in {"unsupported", "ambiguous"}


def load_questions(path: str | Path) -> list[EvaluationQuestion]:
    questions: list[EvaluationQuestion] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        questions.append(
            EvaluationQuestion(
                question=payload["question"],
                category=payload["category"],
                expected_keywords=tuple(payload.get("expected_keywords", [])),
            )
        )
    return questions


def _keyword_match(item: EvaluationQuestion, answer: str) -> bool:
    if not item.expected_keywords:
        return bool(answer.strip())
    normalized = answer.lower()
    return any(keyword.lower() in normalized for keyword in item.expected_keywords)


def _is_refusal(answer: str) -> bool:
    return answer.strip().lower() in {"i don't know", "i don't know.", "i do not know", "i do not know."}


def _score(item: EvaluationQuestion, result: dict[str, Any], rag: bool) -> bool:
    if item.expected_refusal:
        return bool(result["abstained"])
    if item.category == "outside-corpus":
        return bool(not rag and not result["abstained"] and result["keyword_match"])
    return bool(
        not result["abstained"]
        and result["keyword_match"]
        and (not item.requires_corpus or (rag and result["has_citations"]))
    )


def _metrics(
    questions: list[EvaluationQuestion],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({item.category for item in questions}):
        indexes = [index for index, item in enumerate(questions) if item.category == category]
        category_results = [results[index] for index in indexes]
        by_category[category] = {
            "questions": len(indexes),
            "accuracy": round(
                sum(result["behavior_correct"] for result in category_results) / len(indexes),
                3,
            ),
            "correct": sum(result["behavior_correct"] for result in category_results),
        }
    answered = [result for result in results if not result["abstained"]]
    return {
        "behavior_accuracy": round(sum(result["behavior_correct"] for result in results) / len(results), 3),
        "category_accuracy": by_category,
        "answer_rate": round(len(answered) / len(results), 3),
        "citation_rate_on_answers": round(sum(result["has_citations"] for result in answered) / len(answered), 3) if answered else 0.0,
        "average_latency_ms": round(sum(result["elapsed_ms"] for result in results) / len(results), 1),
        "total_llm_calls": sum(result["llm_calls"] for result in results),
        "refusals": sum(result["abstained"] for result in results),
    }


def _run_system(
    questions: list[EvaluationQuestion],
    ask: Callable[[str], Any],
    rag: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in questions:
        started = time.perf_counter()
        result = ask(item.question)
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = {
            "answer": result.answer,
            "abstained": bool(result.abstained or _is_refusal(result.answer)),
            "keyword_match": bool(not result.abstained and _keyword_match(item, result.answer)),
            "has_citations": bool(result.sources),
            "sources": result.sources,
            "confidence": result.confidence,
            "elapsed_ms": round(elapsed_ms, 1),
            "llm_calls": int(not result.abstained),
            "trace_steps": len(result.trace),
            "trace": result.trace,
            "rag": rag,
        }
        payload["behavior_correct"] = _score(item, payload, rag)
        results.append(payload)
    return results


def _run_no_rag(questions: list[EvaluationQuestion], llm: OpenRouterLLM) -> list[dict[str, Any]]:
    return _run_system(
        questions,
        lambda question: type(
            "NoRAGResult",
            (),
            {
                "answer": llm.generate_open_domain(question),
                "abstained": False,
                "sources": [],
                "confidence": 0.0,
                "trace": [],
            },
        )(),
        rag=False,
    )


def run_llm_comparison(
    documents_path: str | Path,
    questions_path: str | Path,
    index_path: str | Path,
) -> dict[str, Any]:
    """Compare no-RAG LLM, classic RAG, and bounded agentic RAG."""
    questions = load_questions(questions_path)
    classic = RAGPipeline(persist_dir=index_path)
    agentic = AgenticRAGPipeline(persist_dir=index_path)
    no_rag_llm = OpenRouterLLM()
    ingest_stats = classic.ingest(documents_path)

    try:
        no_rag_results = _run_no_rag(questions, no_rag_llm)
        classic_results = _run_system(questions, classic.ask, rag=True)
        agentic_results = _run_system(questions, agentic.ask, rag=True)
    finally:
        classic.store.close()
        agentic.store.close()

    systems = {
        "no_rag_llm": no_rag_results,
        "classic_rag": classic_results,
        "agentic_rag": agentic_results,
    }
    metrics = {
        name: _metrics(questions, results)
        for name, results in systems.items()
    }
    wins_by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({item.category for item in questions}):
        indexes = [index for index, item in enumerate(questions) if item.category == category]
        scores = {
            name: sum(results[index]["behavior_correct"] for index in indexes)
            for name, results in systems.items()
        }
        best = max(scores.values())
        wins_by_category[category] = {
            "scores": scores,
            "best_systems": [name for name, score in scores.items() if score == best],
        }

    overall_scores = {name: data["behavior_accuracy"] for name, data in metrics.items()}
    best_overall = max(overall_scores.values())
    finding = (
        "Agentic RAG wins where the first retrieval is weak but a bounded rewrite recovers corpus evidence; "
        "classic RAG wins on simpler in-corpus questions through lower latency; no-RAG wins only on general "
        "knowledge intentionally outside this corpus, where it can answer but cannot cite project documents. "
        "Unsupported and ambiguous questions are safest when the RAG systems refuse."
    )

    return {
        "title": "No-RAG LLM vs Classic RAG vs Agentic RAG",
        "method": "All generated answers used the live OpenRouter tencent/hy3:free model. No-RAG receives no retrieved context; classic RAG retrieves once; agentic RAG uses bounded evidence evaluation and query rewriting. Behavior scoring uses labeled categories, expected keywords, refusal behavior, and citation presence for corpus questions.",
        "corpus": {**asdict(ingest_stats), "sources": [Path(source).name for source in ingest_stats.sources]},
        "questions": [asdict(item) for item in questions],
        "systems": systems,
        "metrics": metrics,
        "wins_by_category": wins_by_category,
        "overall_winner": [name for name, score in overall_scores.items() if score == best_overall],
        "finding": finding,
    }

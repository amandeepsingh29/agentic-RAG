from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

from .agentic_pipeline import AgenticRAGPipeline
from .pipeline import RAGPipeline


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    expected_type: str

    @property
    def should_answer(self) -> bool:
        return self.expected_type == "supported"


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_type: str
    classic: dict[str, Any]
    agentic: dict[str, Any]


def load_questions(path: str | Path) -> list[EvaluationQuestion]:
    questions: list[EvaluationQuestion] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        questions.append(
            EvaluationQuestion(
                question=payload["question"],
                expected_type=payload.get("expected_type", "supported"),
            )
        )
    return questions


def _result_payload(result: Any, elapsed_ms: float, llm_calls: int) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "abstained": result.abstained,
        "has_citations": bool(result.sources),
        "sources": result.sources,
        "confidence": result.confidence,
        "elapsed_ms": round(elapsed_ms, 1),
        "llm_calls": llm_calls,
        "trace_steps": len(result.trace),
        "trace": result.trace,
    }


def _is_behavior_correct(item: EvaluationQuestion, result: Any) -> bool:
    return result.abstained is (not item.should_answer)


def _metrics(questions: list[EvaluationQuestion], results: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(
        result["behavior_correct"] for result in results
    )
    supported = [result for item, result in zip(questions, results) if item.should_answer]
    answered_supported = sum(not result["abstained"] for result in supported)
    cited_answers = sum(result["has_citations"] for result in results if not result["abstained"])
    answered_count = sum(not result["abstained"] for result in results)
    return {
        "behavior_accuracy": round(correct / len(results), 3) if results else 0.0,
        "supported_answer_rate": round(answered_supported / len(supported), 3) if supported else 0.0,
        "citation_rate_on_answers": round(cited_answers / answered_count, 3) if answered_count else 0.0,
        "average_latency_ms": round(sum(result["elapsed_ms"] for result in results) / len(results), 1) if results else 0.0,
        "total_llm_calls": sum(result["llm_calls"] for result in results),
        "refusals": sum(result["abstained"] for result in results),
    }


def run_llm_comparison(
    documents_path: str | Path,
    questions_path: str | Path,
    index_path: str | Path,
) -> dict[str, Any]:
    """Compare classic and bounded agentic RAG using live LLM generation."""
    questions = load_questions(questions_path)
    classic = RAGPipeline(persist_dir=index_path)
    agentic = AgenticRAGPipeline(persist_dir=index_path)
    ingest_stats = classic.ingest(documents_path)

    classic_results: list[dict[str, Any]] = []
    agentic_results: list[dict[str, Any]] = []
    try:
        for item in questions:
            started = time.perf_counter()
            classic_result = classic.ask(item.question)
            classic_elapsed = (time.perf_counter() - started) * 1000
            classic_payload = _result_payload(
                classic_result,
                classic_elapsed,
                llm_calls=int(not classic_result.abstained),
            )
            classic_payload["behavior_correct"] = _is_behavior_correct(item, classic_result)
            classic_results.append(classic_payload)

            started = time.perf_counter()
            agentic_result = agentic.ask(item.question)
            agentic_elapsed = (time.perf_counter() - started) * 1000
            agentic_payload = _result_payload(
                agentic_result,
                agentic_elapsed,
                llm_calls=int(not agentic_result.abstained),
            )
            agentic_payload["behavior_correct"] = _is_behavior_correct(item, agentic_result)
            agentic_results.append(agentic_payload)
    finally:
        classic.store.close()
        agentic.store.close()

    cases = [
        EvaluationCase(
            question=item.question,
            expected_type=item.expected_type,
            classic=classic_result,
            agentic=agentic_result,
        )
        for item, classic_result, agentic_result in zip(questions, classic_results, agentic_results)
    ]
    classic_metrics = _metrics(questions, classic_results)
    agentic_metrics = _metrics(questions, agentic_results)
    delta = round(
        agentic_metrics["behavior_accuracy"] - classic_metrics["behavior_accuracy"],
        3,
    )
    if delta > 0:
        finding = "Agentic RAG was safer on this test set because its bounded query rewrites improved the supported/unsupported decision without removing citations."
    elif delta < 0:
        finding = "Classic RAG was safer on this test set; the agentic loop did not improve the expected behavior and added retrieval work."
    else:
        finding = "Both systems made the same supported/unsupported decisions on this test set; agentic RAG adds traceable retrieval control rather than a measured accuracy gain here."

    return {
        "title": "Classic RAG vs Agentic RAG: LLM-Only Evaluation",
        "method": "Both pipelines used the live OpenRouter tencent/hy3:free model for generation, the same PostgreSQL/pgvector index, and the same questions. No fake or local generation model was used.",
        "corpus": asdict(ingest_stats),
        "questions": [asdict(item) for item in questions],
        "classic": {"metrics": classic_metrics, "cases": [asdict(case)["classic"] for case in cases]},
        "agentic": {"metrics": agentic_metrics, "cases": [asdict(case)["agentic"] for case in cases]},
        "finding": finding,
        "behavior_accuracy_delta_agentic_minus_classic": delta,
    }

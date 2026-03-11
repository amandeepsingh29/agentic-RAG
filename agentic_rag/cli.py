from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agentic_pipeline import AgenticRAGPipeline
from .evaluation import run_llm_comparison
from .ingestion import download_kubernetes_docs, load_documents
from .pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-docs")
    download.add_argument("--output", default="data/kubernetes")

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--documents", default="data/kubernetes")
    ingest.add_argument("--index", default="data/pgvector")

    ask = subparsers.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--index", default="data/pgvector")
    ask.add_argument("--mode", choices=("agentic", "classic"), default="agentic")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--documents", default="data/kubernetes")
    eval_parser.add_argument("--questions", default="data/eval_queries.jsonl")
    eval_parser.add_argument("--index", default="data/pgvector")
    eval_parser.add_argument("--mode", choices=("agentic", "classic"), default="agentic")

    compare = subparsers.add_parser("compare", help="Compare classic and agentic RAG with live LLM generation")
    compare.add_argument("--documents", default="data/kubernetes")
    compare.add_argument("--questions", default="data/eval_queries.jsonl")
    compare.add_argument("--index", default="data/pgvector")
    compare.add_argument("--output", default="data/evaluation_report.json")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "compare":
        report = run_llm_comparison(args.documents, args.questions, args.index)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "finding": report["finding"], "classic": report["classic"]["metrics"], "agentic": report["agentic"]["metrics"]}, indent=2))
        return

    if args.command == "download-docs":
        written = download_kubernetes_docs(args.output)
        print(json.dumps({"downloaded": [str(path) for path in written]}, indent=2))
        return

    pipeline_class = AgenticRAGPipeline if getattr(args, "mode", "classic") == "agentic" else RAGPipeline
    pipeline = pipeline_class(persist_dir=args.index)

    if args.command == "ingest":
        stats = pipeline.ingest(args.documents)
        print(json.dumps(stats.__dict__, indent=2))
        return

    if args.command == "ask":
        result = pipeline.ask(args.question)
        print(json.dumps(result.__dict__, indent=2))
        return

    if args.command == "eval":
        stats = pipeline.ingest(args.documents)
        print(json.dumps({"ingest": stats.__dict__}, indent=2))
        questions_path = Path(args.questions)
        if questions_path.exists():
            for line in questions_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                result = pipeline.ask(payload["question"])
                print(json.dumps({"question": payload["question"], "result": result.__dict__}, indent=2))


if __name__ == "__main__":
    main()

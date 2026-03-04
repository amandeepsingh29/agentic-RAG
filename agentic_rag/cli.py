from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answering import AnswerEngine
from .ingestion import load_documents
from .retrieval import HybridRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("paths", nargs="+")

    ask = subparsers.add_parser("ask")
    ask.add_argument("corpus")
    ask.add_argument("question")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        documents = load_documents(args.paths)
        retriever = HybridRetriever()
        chunks = retriever.index_documents(documents)
        print(json.dumps({"documents": len(documents), "chunks": len(chunks)}, indent=2))
        return

    if args.command == "ask":
        corpus_path = Path(args.corpus)
        documents = load_documents([corpus_path])
        retriever = HybridRetriever()
        retriever.index_documents(documents)
        result = AnswerEngine().answer(args.question, retriever.retrieve(args.question))
        print(json.dumps(result.__dict__, indent=2))


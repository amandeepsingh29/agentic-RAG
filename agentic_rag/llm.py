from __future__ import annotations

from dataclasses import dataclass
import os

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OpenRouterLLM:
    model_name: str = "tencent/hy3:free"
    api_key: str | None = None
    referer: str | None = None
    title: str | None = None
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        self.referer = self.referer or os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
        self.title = self.title or os.getenv("OPENROUTER_TITLE", "RAG Documentation Assistant")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for generation.")

    def generate(self, question: str, context: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a documentation assistant. Answer only from the provided context. "
                        "If the context does not contain the answer, reply exactly: I don't know. "
                        "Keep the answer concise and grounded."
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_prompt(question, context),
                },
            ],
            "temperature": 0,
            "top_p": 1,
        }
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.referer,
                "X-OpenRouter-Title": self.title,
            },
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        return (
            "Use the context below to answer the question. "
            'If the answer is not present, respond exactly with: "I don\'t know."\\n\\n'
            f"Question: {question}\\n\\n"
            f"Context:\\n{context}\\n\\n"
            "Answer:"
        )

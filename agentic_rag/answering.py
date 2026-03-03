from __future__ import annotations

from .models import AnswerResult, RetrievalHit
from .policy import should_abstain


class AnswerEngine:
    def answer(self, question: str, hits: list[RetrievalHit]) -> AnswerResult:
        abstained, reason = should_abstain(hits, question)
        if abstained:
            return AnswerResult(
                question=question,
                answer="I couldn’t find enough support in the available documents to answer safely.",
                citations=[],
                confidence=0.0,
                abstained=True,
                refusal_reason=reason,
            )

        selected_hits = hits[:3]
        answer_lines = []
        citations = []
        for hit in selected_hits:
            sentence = hit.chunk.text.split(".")[0].strip()
            if sentence:
                answer_lines.append(sentence)
            citations.append(
                {
                    "source": hit.chunk.source,
                    "chunk_id": hit.chunk.chunk_id,
                    "score": round(hit.score, 4),
                    "snippet": hit.chunk.text[:240],
                }
            )

        unique_lines = []
        for line in answer_lines:
            if line not in unique_lines:
                unique_lines.append(line)
        answer = " ".join(unique_lines).strip()
        confidence = min(0.99, sum(hit.score for hit in selected_hits) / max(1, len(selected_hits)))

        if not answer:
            return AnswerResult(
                question=question,
                answer="I couldn’t synthesize a grounded answer from the available evidence.",
                citations=citations,
                confidence=float(confidence),
                abstained=True,
                refusal_reason="No extractable grounded sentence was found.",
            )

        return AnswerResult(
            question=question,
            answer=answer,
            citations=citations,
            confidence=float(confidence),
            abstained=False,
        )


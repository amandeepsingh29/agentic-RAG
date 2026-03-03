from __future__ import annotations

from .models import RetrievalHit


def should_abstain(hits: list[RetrievalHit], question: str, minimum_score: float = 0.18) -> tuple[bool, str | None]:
    if not hits:
        return True, "No relevant evidence was retrieved."

    if hits[0].score < minimum_score:
        return True, "Retrieved evidence was too weak to answer safely."

    question_terms = {term.lower() for term in question.split() if len(term) > 3}
    evidence_text = " ".join(hit.chunk.text.lower() for hit in hits[:3])
    overlap = sum(1 for term in question_terms if term in evidence_text)
    if question_terms and overlap == 0:
        return True, "The retrieved context did not cover the question terms."

    return False, None


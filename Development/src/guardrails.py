import re
from config import DEBUG
from src.latency_tracker import track_latency

_WORD_RE = re.compile(r"[a-zA-Zऀ-ॿ஀-௿]+")


def _tokenize(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


class Guardrails:
    """Grounding and answer-quality checks."""

    @track_latency("grounding_check")
    def check_grounding(self, answer: str, retrieved_docs: list) -> float:
        """Score how well the answer is supported by the retrieved documents (word overlap)."""
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0

        context_tokens = set()
        for doc in retrieved_docs:
            context_tokens |= _tokenize(doc)

        if not context_tokens:
            return 0.0

        overlap = answer_tokens & context_tokens
        score = len(overlap) / len(answer_tokens)

        if DEBUG:
            print(f"[check_grounding] Score: {score:.2f}")

        return round(min(score, 1.0), 4)

    def validate_answer(self, answer: str) -> bool:
        """Reject empty or degenerate answers."""
        if not answer or len(answer.strip()) == 0:
            return False
        if len(answer.strip()) < 3:
            return False
        return True


if __name__ == "__main__":
    guardrails = Guardrails()
    score = guardrails.check_grounding(
        answer="Aadhaar is a unique identity number.",
        retrieved_docs=["Aadhaar is a 12-digit unique identity number issued to Indian residents."]
    )
    print(f"Grounding score: {score}")
    print(f"Valid: {guardrails.validate_answer('Aadhaar is a unique identity number.')}")

import math
import re

import torch

from config import DEBUG
from src.latency_tracker import track_latency

_WORD_RE = re.compile(r"[a-zA-Zऀ-ॿ஀-௿]+")

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Refusal/non-answer phrases that indicate the model didn't actually answer.
_REFUSAL_PHRASES = [
    "i don't know", "i do not know", "unknown", "not available",
    "i could not find", "i cannot find", "no information",
]


def _tokenize(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text or "")}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Guardrails:
    """Grounding and answer-quality checks.

    Grounding uses a multilingual cross-encoder (trained on mMARCO - the same
    corpus family as our MSMARCO-XI index) to score how well the answer is
    supported by each retrieved passage. Word overlap breaks down badly for
    morphologically rich languages (Hindi case suffixes, etc.) and for
    paraphrased LLM output, so it's kept only as a fallback if the
    cross-encoder can't be loaded.
    """

    def __init__(self):
        self.cross_encoder = None
        try:
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
            if DEBUG:
                print(f"[Guardrails] Loaded cross-encoder: {CROSS_ENCODER_MODEL} on {device}")
        except Exception as e:
            if DEBUG:
                print(f"[Guardrails] Cross-encoder unavailable ({e}), falling back to word overlap")

    @track_latency("grounding_check")
    def check_grounding(self, answer: str, retrieved_docs: list) -> float:
        """Score how well the answer is supported by the retrieved documents."""
        if not answer or not answer.strip() or not retrieved_docs:
            return 0.0

        if self.cross_encoder is not None:
            score = self._check_grounding_cross_encoder(answer, retrieved_docs)
        else:
            score = self._check_grounding_word_overlap(answer, retrieved_docs)

        if DEBUG:
            print(f"[check_grounding] Score: {score:.4f}")

        return round(score, 4)

    def _check_grounding_cross_encoder(self, answer: str, retrieved_docs: list) -> float:
        pairs = [(answer, doc) for doc in retrieved_docs if doc]
        if not pairs:
            return 0.0
        raw_scores = self.cross_encoder.predict(pairs)
        best = max(raw_scores)
        return min(_sigmoid(float(best)), 1.0)

    def _check_grounding_word_overlap(self, answer: str, retrieved_docs: list) -> float:
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0

        context_tokens = set()
        for doc in retrieved_docs:
            context_tokens |= _tokenize(doc)

        if not context_tokens:
            return 0.0

        overlap = answer_tokens & context_tokens
        return min(len(overlap) / len(answer_tokens), 1.0)

    def validate_answer(self, answer: str) -> bool:
        """Reject empty, too-short, or refusal-style non-answers."""
        if not answer or len(answer.strip()) == 0:
            return False
        if len(answer.strip()) < 3:
            return False
        lowered = answer.strip().lower()
        if any(phrase in lowered for phrase in _REFUSAL_PHRASES):
            return False
        return True


if __name__ == "__main__":
    guardrails = Guardrails()
    score = guardrails.check_grounding(
        answer="A corporation is a business entity chartered by a state.",
        retrieved_docs=["A corporation is the most common form of business organization, "
                         "chartered by a state and given legal rights separate from its owners."]
    )
    print(f"Grounding score: {score}")
    print(f"Valid: {guardrails.validate_answer('A corporation is a business entity.')}")
    refusal_text = "I don't know the answer."
    print(f"Valid (refusal): {guardrails.validate_answer(refusal_text)}")

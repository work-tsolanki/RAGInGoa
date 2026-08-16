import math
import re

import numpy as np
import torch

from config import DEBUG
from src.latency_tracker import track_latency

_WORD_RE = re.compile(r"[a-zA-Zऀ-ॿ஀-௿]+")

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Reference set for the off-topic gate (see check_off_topic). The corpus
# (MSMARCO-XI) is open-domain general-knowledge QA, not a narrow business
# topic - there's no bounded "list of in-scope subjects" to enumerate. These
# span the actual breadth observed across this project's own benchmark runs
# (government/legal process, business/tax, geography/travel, food/culture,
# science/medicine, history, pop culture) rather than one narrow theme, plus
# a few Hindi/Gujarati anchors since the multilingual embedding space isn't
# perfectly language-symmetric (a real, measured gap - see the Aug 2026
# percentile-batch hedge-rate investigation). New reference queries can be
# added here if a genuine in-scope query starts getting misflagged.
_OFF_TOPIC_REFERENCE_QUERIES = [
    "How to apply for a passport",
    "What is a corporation",
    "income tax filing deadline",
    "voter ID card requirements",
    "how does GST registration work",
    "what is Goa famous for",
    "what is kaju katli",
    "what is barter trade",
    "tell me about forts in India",
    "historical sites near Chichen Itza",
    "describe the Udaipur lake palace",
    "what are New Orleans beignets",
    "what is trigger finger condition",
    "diesel fuel cost per gallon",
    "standard molar entropy in chemistry",
    "what is cornflour made from",
    "average high school teacher salary",
    "who created the Iron Man comic character",
    "what degree do you need to become a teacher",
    "death's head insignia military history",
    "what is a DBA business registration",
    "how is an S-Corp taxed",
    "beaches in Goa",
    "what is feni liquor made from",
    "GST tax credit claims process",
    "documents needed for a passport card",
    "Portuguese architecture in Goa",
    "bilateral trade vs multilateral trade",
    "significance of Isthmia in Greek mythology",
    "who was Harriet Tubman",
    "पासपोर्ट के लिए आवेदन कैसे करें",
    "गोवा किस लिए प्रसिद्ध है",
    "મતદાર ID માટે શું જરૂરી છે",
    "કંપની એટલે શું",
    "ફેની શું છે",
    "કાજુ કતલી શું છે",
]

# Calibrated (see scripts/calibrate_off_topic_threshold.py) against the
# project's own real benchmark queries (must NOT trigger) vs deliberately
# off-topic ones - creative writing, pure computation, casual chat, meta
# questions about the assistant itself (must trigger). Sits between the two
# observed similarity distributions, not chosen by eye.
OFF_TOPIC_SIMILARITY_THRESHOLD = 0.499  # midpoint between max(off-topic)=0.4123 and
                                         # min(in-scope)=0.5857, per scripts/calibrate_off_topic_threshold.py

# Minimum-viable unsafe-input gate: obvious-intent phrase patterns, not a
# full classifier. This is a different failure mode than "ungrounded
# answer" (check_grounding) - it's "shouldn't have been processed as a real
# query at all," so it runs before retrieval/generation even starts.
_UNSAFE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bhow (?:do|can|to) i? ?(?:make|build|create) (?:a )?(?:bomb|explosive|weapon)",
        r"\bhow (?:do|can|to) i? ?(?:hurt|kill|harm) (?:myself|someone|others)",
        r"\b(?:suicide|self.?harm) (?:methods|instructions|how.?to)",
        r"\bhow (?:do|can|to) i? ?(?:synthesize|make) (?:illegal drugs|meth|cocaine)",
        r"\bchild (?:sexual|porn|abuse)",
    ]
]

_UNSAFE_MESSAGES = {
    "en": "I can't help with that request.",
    "hi": "मैं इस अनुरोध में मदद नहीं कर सकता।",
    "gu": "હું આ વિનંતીમાં મદદ કરી શકતો નથી.",
}

_OFF_TOPIC_MESSAGES = {
    "en": "That looks outside what this system can help with - it's built to answer questions grounded in its own document corpus. Try rephrasing as a factual question, or ask about a related topic.",
    "hi": "यह इस सिस्टम की सहायता क्षेत्र से बाहर लगता है - यह अपने दस्तावेज़ों पर आधारित प्रश्नों के उत्तर देने के लिए बनाया गया है। कृपया इसे एक तथ्यात्मक प्रश्न के रूप में दोबारा पूछें।",
    "gu": "આ આ સિસ્ટમની મદદના ક્ષેત્રની બહાર લાગે છે - તે તેના પોતાના દસ્તાવેજો પર આધારિત પ્રશ્નોના જવાબ આપવા માટે બનાવવામાં આવ્યું છે. કૃપા કરીને તેને તથ્યાત્મક પ્રશ્ન તરીકે ફરીથી પૂછો.",
}


def build_unsafe_response(language: str = "en") -> str:
    return _UNSAFE_MESSAGES.get(language, _UNSAFE_MESSAGES["en"])


def build_off_topic_response(language: str = "en") -> str:
    return _OFF_TOPIC_MESSAGES.get(language, _OFF_TOPIC_MESSAGES["en"])


def check_unsafe(query_text: str) -> bool:
    """True if the query matches an obvious-intent unsafe pattern. Runs
    ahead of everything else - retrieval/generation cost shouldn't be spent
    on something that should never have been processed as a real query."""
    if not query_text:
        return False
    return any(p.search(query_text) for p in _UNSAFE_PATTERNS)

# Refusal/non-answer phrases that indicate the model didn't actually answer.
_REFUSAL_PHRASES = [
    "i don't know", "i do not know", "unknown", "not available",
    "i could not find", "i cannot find", "no information",
]

# Hedge message for the confident-hallucination case: the answer passed
# validate_answer() (not empty, not a refusal phrase) but scored below
# ANSWER_CACHE_MIN_GROUNDING - the retrieved context doesn't actually
# support it. Distinct from validate_answer's own fallback, which covers
# the model explicitly saying it doesn't know. Only en/hi/gu are hand-
# verified; every other language code falls back to English rather than
# risk shipping an unverified machine translation in a hedge message.
_LOW_GROUNDING_MESSAGES = {
    "en": "I couldn't find information in the available sources that confidently answers this question. Could you rephrase it, or ask about a related topic?",
    "hi": "मुझे उपलब्ध जानकारी में इस प्रश्न का विश्वसनीय उत्तर नहीं मिला। कृपया प्रश्न को दोबारा पूछें या किसी संबंधित विषय के बारे में पूछें।",
    "gu": "મને ઉપલબ્ધ માહિતીમાં આ પ્રશ્નનો વિશ્વસનીય જવાબ મળ્યો નથી. કૃપા કરીને પ્રશ્નને ફરીથી પૂછો અથવા સંબંધિત વિષય વિશે પૂછો.",
}


def build_low_grounding_response(language: str = "en") -> str:
    """Language-aware hedge message - reuses the language code already
    threaded through the pipeline (target_language/request.language)
    rather than re-detecting it from the query text."""
    return _LOW_GROUNDING_MESSAGES.get(language, _LOW_GROUNDING_MESSAGES["en"])


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

    def __init__(self, embedding_service=None):
        self.cross_encoder = None
        try:
            from sentence_transformers import CrossEncoder
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
            # Same lazy-device quirk as SentenceTransformer: __init__ only
            # records _target_device, the actual .to(device) transfer
            # happens on first predict() otherwise - see embedding_service.py.
            self.cross_encoder.model.to(device)
            if DEBUG:
                print(f"[Guardrails] Loaded cross-encoder: {CROSS_ENCODER_MODEL} on {device}")
        except Exception as e:
            if DEBUG:
                print(f"[Guardrails] Cross-encoder unavailable ({e}), falling back to word overlap")

        # Off-topic gate needs the same embedding model/space the rest of the
        # pipeline already uses (so it can score the caller's precomputed
        # query_embedding directly, no re-embedding). Reuses the shared
        # EmbeddingService instance rather than loading a second copy of the
        # model - optional so this class stays usable standalone (see
        # __main__ below) without the full service graph.
        self._reference_embeddings = None
        if embedding_service is not None:
            self._reference_embeddings = np.stack([
                embedding_service.embed_query(q) for q in _OFF_TOPIC_REFERENCE_QUERIES
            ])

    def check_off_topic(self, query_embedding) -> tuple:
        """Returns (is_off_topic, max_similarity). Compares the query's
        embedding against a fixed reference set via cosine similarity - if
        it isn't close to anything the corpus is actually about, decline
        before spending retrieval/generation cost on it. Falls back to
        never flagging (False, 1.0) if no embedding_service was provided at
        construction time, rather than blocking every query."""
        if self._reference_embeddings is None:
            return False, 1.0
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        refs = self._reference_embeddings / (
            np.linalg.norm(self._reference_embeddings, axis=1, keepdims=True) + 1e-9
        )
        similarities = refs @ q
        max_sim = float(np.max(similarities))
        return max_sim < OFF_TOPIC_SIMILARITY_THRESHOLD, max_sim

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

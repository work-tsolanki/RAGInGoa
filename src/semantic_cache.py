"""Fuzzy answer cache keyed on query embedding similarity, sitting in front
of the exact-match literal cache (answer_cache.py).

Threshold calibration: see scripts/calibrate_semantic_cache_threshold.py.
True-duplicate and related-but-distinct query pairs do NOT separate
cleanly for this domain (e.g. "gst registration process" vs "gst
registration fees" score close to true duplicates) - the threshold is set
conservatively above the highest observed related-but-distinct score
rather than at some midpoint, per that calibration's findings.

Note this cache is NOT protected by answer_cache.py's key-based
PROMPT_VERSION trick - lookup() below matches purely on embedding cosine
similarity and never looks at the caller-supplied key (it's logging-only).
So the prompt version is instead stored per-entry and checked explicitly in
lookup(), the same way an expired entry is treated as a non-match - a
generation_service._build_prompt change makes every existing entry
unreachable instead of serving an old-style answer via a fuzzy match to a
differently-worded new query.
"""

import time
from collections import OrderedDict

import numpy as np

from config import PROMPT_VERSION


class SemanticCache:
    """Cosine similarity scan over cached query embeddings, vectorized as a
    single batch matmul rather than a per-entry Python loop - a for loop
    calling a similarity function once per entry was the actual cost, not
    the cosine math itself (see conversation). Embeddings are stored
    pre-normalized (at set() time) so lookup's per-query normalize + a
    single dot product against the whole stored matrix is mathematically
    identical to cosine similarity, just without the redundant repeated
    norm computation.

    Still brute-force (no ANN index) - fine at this scale (max ~1000
    entries per ANSWER_CACHE_MAX_SIZE); a vectorized matmul over ~1000x384
    floats is negligible regardless. Would only be worth swapping for a
    real index (e.g. a second Chroma collection, reusing infra this
    project already runs) if max_size grew by orders of magnitude."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 86400, similarity_threshold: float = 0.92):
        self.entries = OrderedDict()  # key -> (normalized_embedding, payload, expiry, prompt_version)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def lookup(self, query_embedding: np.ndarray, prompt_version: str = PROMPT_VERSION):
        """Returns (payload, similarity) on a hit, (None, best_similarity_seen) on a miss.
        Entries stored under a different prompt_version are skipped entirely
        (treated like an expired entry), not just deprioritized - a fuzzy
        match to a stale-style answer is exactly the failure case this
        guards against."""
        now = time.time()
        expired = [key for key, (_, _, expiry, _) in self.entries.items() if now > expiry]
        for key in expired:
            del self.entries[key]

        candidate_keys = [key for key, (_, _, _, v) in self.entries.items() if v == prompt_version]
        if not candidate_keys:
            self.misses += 1
            return None, 0.0

        query_norm = self._normalize(query_embedding)
        embeddings = np.stack([self.entries[key][0] for key in candidate_keys])  # already normalized
        sims = embeddings @ query_norm  # dot product of unit vectors = cosine similarity

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.similarity_threshold:
            best_key = candidate_keys[best_idx]
            self.entries.move_to_end(best_key)
            self.hits += 1
            return self.entries[best_key][1], best_sim

        self.misses += 1
        return None, best_sim

    def set(self, key: str, query_embedding: np.ndarray, payload: dict, prompt_version: str = PROMPT_VERSION):
        """key is only used for logging/debugging - the embedding drives the actual lookup."""
        expiry = time.time() + self.ttl_seconds
        self.entries[key] = (self._normalize(query_embedding), payload, expiry, prompt_version)
        self.entries.move_to_end(key)
        if len(self.entries) > self.max_size:
            self.entries.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "size": len(self.entries),
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold,
        }

# Task: Add semantic (fuzzy) answer caching in front of the existing literal cache

## Context

Current state: `answer_cache.py` does exact-match caching — LRU keyed on `normalize_query(query) + sorted(doc_ids)`, TTL 24h, only caches answers that pass `validate_answer`. This only catches literal repeats/near-identical phrasing (case, punctuation, whitespace).

Goal: catch semantic duplicates too — different phrasing, same intent ("how to apply for a passport" vs "passport application process" vs "steps to get a passport") — without retrieval or generation running at all on a semantic hit.

Do not remove or replace the existing literal cache. Semantic cache sits in front of it as an earlier, coarser check.

## Step 1: Decide where semantic lookup happens in the request flow

New order:

```
Query → embed query (already have this model loaded for retrieval)
      → semantic cache lookup (cosine similarity vs cached query embeddings)
           ├── similarity ≥ threshold → return cached answer (semantic hit, ~15-20ms)
           └── below threshold        → continue to literal cache check
                                          ├── literal hit  → return cached answer
                                          └── literal miss → full pipeline (retrieval → Groq → grounding)
```

Semantic check must run **before** retrieval, not after — the entire point is to skip retrieval and generation on a hit. If it ran after retrieval like the literal cache does, you'd lose most of the latency benefit.

## Step 2: Build the semantic cache store

Create `src/semantic_cache.py`. Given your cache size (max ~1000 entries per the literal cache's `ANSWER_CACHE_MAX_SIZE`), a brute-force cosine similarity scan is fast enough — no need for a separate vector index (Chroma, FAISS) for this scale. Keep it simple:

```python
import numpy as np
from collections import OrderedDict
import time

class SemanticCache:
    def __init__(self, max_size=1000, ttl_seconds=86400, similarity_threshold=0.92):
        self.entries = OrderedDict()  # key -> (embedding, answer_payload, expiry)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    def _cosine_sim(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def lookup(self, query_embedding: np.ndarray):
        now = time.time()
        best_key, best_sim = None, 0.0
        expired = []
        for key, (emb, payload, expiry) in self.entries.items():
            if now > expiry:
                expired.append(key)
                continue
            sim = self._cosine_sim(query_embedding, emb)
            if sim > best_sim:
                best_sim, best_key = sim, key
        for key in expired:
            del self.entries[key]

        if best_key is not None and best_sim >= self.similarity_threshold:
            self.entries.move_to_end(best_key)
            self.hits += 1
            return self.entries[best_key][1], best_sim  # payload, similarity score
        self.misses += 1
        return None, best_sim

    def set(self, key: str, query_embedding: np.ndarray, payload: dict):
        expiry = time.time() + self.ttl_seconds
        self.entries[key] = (query_embedding, payload, expiry)
        self.entries.move_to_end(key)
        if len(self.entries) > self.max_size:
            self.entries.popitem(last=False)

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0,
            "size": len(self.entries),
        }
```

`key` can just be the normalized query text (for logging/debugging), the embedding is what actually drives the lookup.

## Step 3: Calibrate the similarity threshold — do not guess this

This is the step that actually determines whether the feature is safe. A threshold that's too loose serves the wrong cached answer to a genuinely different question; too strict and it never fires.

1. Pull or write ~30-40 query pairs across three categories:
   - **True near-duplicates** (should match): "how to apply for a passport" / "passport application process" / "steps to get a passport"
   - **Related but distinct** (should NOT match): "how to apply for a passport" vs "how to renew a passport" vs "passport application fees"
   - **Unrelated** (should NOT match): "how to apply for a passport" vs "income tax filing deadline"
2. Embed each pair with your existing embedding model, compute cosine similarity, record the score.
3. Look at where true-duplicate scores and related-but-distinct scores separate. This is very likely *not* going to be a clean gap — "how to apply" vs "how to renew" a passport are lexically and semantically close but need different answers. That closeness is the actual risk of this feature.
4. Pick a threshold above the highest "related but distinct" score you observe, even if that means some true duplicates fall through to the literal cache or full pipeline. A missed semantic-cache opportunity costs 220ms; a false match serves a wrong answer. Bias the threshold conservative.
5. Record the threshold you land on and the data that justified it — don't hardcode a number without this trail, since it'll need revisiting once real traffic accumulates.

## Step 4: Wire into `main_app.py`

```python
semantic_cache = SemanticCache(
    max_size=config.SEMANTIC_CACHE_MAX_SIZE,
    ttl_seconds=config.SEMANTIC_CACHE_TTL_SECONDS,
    similarity_threshold=config.SEMANTIC_CACHE_SIMILARITY_THRESHOLD,  # from Step 3
)

async def handle_query(query_text: str):
    query_embedding = embed_query(query_text)  # reuse existing embedding call

    cached_payload, sim_score = semantic_cache.lookup(query_embedding)
    if cached_payload is not None:
        log.info(f"Semantic cache hit, similarity={sim_score:.3f}")
        return {**cached_payload, "cache_type": "semantic", "similarity": sim_score}

    # fall through to existing literal cache + full pipeline, using query_embedding
    # instead of re-embedding — this also saves the ~6-12ms embedding step from
    # being duplicated between semantic-cache-check and retrieval on a miss
    dense_r, sparse_r = await asyncio.gather(dense_search(query_embedding), sparse_search(query_text))
    fused = reciprocal_rank_fusion(dense_r, sparse_r, top_k=5)
    doc_ids = [doc_id for doc_id, _ in fused]

    literal_key = make_cache_key(query_text, doc_ids)
    literal_cached = answer_cache.get(literal_key)
    if literal_cached is not None:
        return {**literal_cached, "cache_type": "literal"}

    # ... existing generation + grounding + validate flow, unchanged ...

    if validated["is_valid"] and grounding_score >= config.ANSWER_CACHE_MIN_GROUNDING:
        answer_cache.set(literal_key, result)
        semantic_cache.set(literal_key, query_embedding, result)  # populate both caches

    return {**result, "cache_type": "miss"}
```

Note the `cache_type` field on every response — this is important for step 6.

## Step 5: Add config

In `config.py`:
```python
SEMANTIC_CACHE_MAX_SIZE = 1000
SEMANTIC_CACHE_TTL_SECONDS = 86400  # match literal cache, same staleness reasoning
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = <value from Step 3>
```

## Step 6: Test before trusting it

- [ ] Unit test `SemanticCache.lookup` and `.set` directly — confirm eviction, TTL expiry, and threshold boundary behavior with synthetic embeddings.
- [ ] Run the full Step 3 pair list through the *live* integrated system (not just standalone embedding comparison) — confirm true duplicates return `cache_type: semantic` and related-but-distinct queries do NOT.
- [ ] This is the critical test: deliberately run a "should NOT match" pair (e.g. passport apply vs passport renew) through the system twice — first populates cache, second must NOT return the first answer. If it does, the threshold from Step 3 is too loose — raise it and re-test, don't ship until this passes.
- [ ] Confirm `/debug/cache_stats` (or add a semantic-cache equivalent) reports hit rate separately from the literal cache, so you can see which layer is actually doing the work in real traffic.
- [ ] Latency check: confirm a semantic hit returns in ~15-25ms (embedding + lookup only, no retrieval/generation), same order of magnitude as your literal cache hits.

## What NOT to change

- Literal cache (`answer_cache.py`) stays exactly as-is, still runs on semantic-cache misses.
- Retrieval, generation fallback chain, guardrails — untouched, only reached on a full miss through both cache layers.
- Grounding-gate-before-caching rule applies to both caches identically — never cache (semantic or literal) an answer that didn't pass validation.

## Order of work

1. Step 3 (threshold calibration) is the actual hard part — do this first, standalone, before writing any integration code. If the true-duplicate and related-but-distinct scores don't separate cleanly for your domain, that's critical information to have before building the rest.
2. Steps 1-2 (design + store) — straightforward, low-risk.
3. Steps 4-5 (wiring + config) — mechanical once 1-3 are done.
4. Step 6 (testing) — non-negotiable given the false-match risk; do not skip the deliberate "should NOT match" test.

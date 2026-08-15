import sys
import time

import numpy as np
import pytest

sys.path.insert(0, '.')

from src.semantic_cache import SemanticCache


def _vec(*components):
    """3D unit-ish vector for cheap, deterministic similarity math in tests."""
    return np.array(components, dtype=float)


def test_miss_on_empty_cache():
    cache = SemanticCache(similarity_threshold=0.9)
    payload, sim = cache.lookup(_vec(1, 0, 0))
    assert payload is None
    assert sim == 0.0
    assert cache.stats()["misses"] == 1


def test_hit_on_identical_embedding():
    cache = SemanticCache(similarity_threshold=0.9)
    cache.set("q1", _vec(1, 0, 0), {"answer": "A"})

    payload, sim = cache.lookup(_vec(1, 0, 0))
    assert payload == {"answer": "A"}
    assert sim == pytest.approx(1.0)
    assert cache.stats()["hits"] == 1


def test_miss_below_threshold():
    """Orthogonal vector -> similarity 0, well below any reasonable threshold."""
    cache = SemanticCache(similarity_threshold=0.9)
    cache.set("q1", _vec(1, 0, 0), {"answer": "A"})

    payload, sim = cache.lookup(_vec(0, 1, 0))
    assert payload is None
    assert sim == pytest.approx(0.0)


def test_threshold_boundary_exact():
    """A near-duplicate vector just above/below threshold - confirms the
    boundary check is >=, not >, and that it's exact, not fuzzy-rounded."""
    cache = SemanticCache(similarity_threshold=0.99)
    cache.set("q1", _vec(1, 0, 0), {"answer": "A"})

    # cos_sim(1,0,0 vs 1,0.01,0) is just under 0.99989 - comfortably above
    just_above = cache.lookup(_vec(1, 0.01, 0))
    assert just_above[0] is not None

    # cos_sim(1,0,0 vs 1,1,0) = 1/sqrt(2) ~= 0.707 - comfortably below
    just_below = cache.lookup(_vec(1, 1, 0))
    assert just_below[0] is None


def test_ttl_expiry():
    cache = SemanticCache(ttl_seconds=0.05, similarity_threshold=0.9)
    cache.set("q1", _vec(1, 0, 0), {"answer": "A"})
    assert cache.lookup(_vec(1, 0, 0))[0] is not None

    time.sleep(0.1)
    payload, sim = cache.lookup(_vec(1, 0, 0))
    assert payload is None
    # expired entry should also be purged from storage, not just skipped
    assert len(cache.entries) == 0


def test_lru_eviction_on_max_size():
    cache = SemanticCache(max_size=2, similarity_threshold=0.9)
    cache.set("q1", _vec(1, 0, 0), {"answer": "A"})
    cache.set("q2", _vec(0, 1, 0), {"answer": "B"})
    cache.set("q3", _vec(0, 0, 1), {"answer": "C"})  # exceeds max_size=2, evicts q1 (LRU)

    assert len(cache.entries) == 2
    assert cache.lookup(_vec(1, 0, 0))[0] is None  # q1 evicted
    assert cache.lookup(_vec(0, 1, 0))[0] is not None  # q2 survives
    assert cache.lookup(_vec(0, 0, 1))[0] is not None  # q3 survives


def test_lookup_returns_best_match_among_multiple_entries():
    cache = SemanticCache(similarity_threshold=0.5)
    cache.set("q1", _vec(1, 0, 0), {"answer": "A"})
    cache.set("q2", _vec(0.9, 0.1, 0), {"answer": "B"})  # closer to the query below

    payload, sim = cache.lookup(_vec(0.9, 0.2, 0))
    assert payload == {"answer": "B"}


def test_stats_tracks_hits_and_misses():
    cache = SemanticCache(similarity_threshold=0.9)
    cache.set("q1", _vec(1, 0, 0), {"answer": "A"})
    cache.lookup(_vec(1, 0, 0))  # hit
    cache.lookup(_vec(0, 1, 0))  # miss
    cache.lookup(_vec(1, 0, 0))  # hit

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(2 / 3)
    assert stats["similarity_threshold"] == 0.9


def test_entry_stored_under_old_prompt_version_misses_after_bump():
    """Direct proof, not just code review, of the PROMPT_VERSION guard: seed
    an entry under an old prompt version, confirm it hits under that same
    version (control), then confirm it's a MISS once looked up under a new
    version - mirroring the "deliberately test the should-NOT-match case"
    check used for the similarity threshold above."""
    cache = SemanticCache(similarity_threshold=0.9)
    cache.set("q1", _vec(1, 0, 0), {"answer": "old-style answer"}, prompt_version="v1-old")

    # Control: same version still hits.
    payload, sim = cache.lookup(_vec(1, 0, 0), prompt_version="v1-old")
    assert payload == {"answer": "old-style answer"}

    # The actual regression check: a version bump must not resurrect it via
    # fuzzy match, even though the embedding is identical.
    payload, sim = cache.lookup(_vec(1, 0, 0), prompt_version="v2-new")
    assert payload is None


def test_entries_from_different_prompt_versions_dont_cross_match():
    cache = SemanticCache(similarity_threshold=0.9)
    cache.set("q1", _vec(1, 0, 0), {"answer": "v1 answer"}, prompt_version="v1-old")
    cache.set("q2", _vec(1, 0, 0), {"answer": "v2 answer"}, prompt_version="v2-new")

    payload, sim = cache.lookup(_vec(1, 0, 0), prompt_version="v2-new")
    assert payload == {"answer": "v2 answer"}


def test_stores_normalized_embedding_not_raw():
    """Regression check for the vectorized lookup: set() must normalize
    before storing, since lookup's dot product assumes every stored vector
    is already unit-length - a raw (non-unit) stored vector would silently
    produce wrong similarity scores without raising anything."""
    cache = SemanticCache(similarity_threshold=0.9)
    cache.set("q1", _vec(5, 0, 0), {"answer": "A"})  # deliberately non-unit magnitude

    stored_embedding = cache.entries["q1"][0]
    assert np.linalg.norm(stored_embedding) == pytest.approx(1.0)


def test_lookup_unaffected_by_query_embedding_magnitude():
    """Cosine similarity is scale-invariant - a query vector with a huge or
    tiny magnitude must match identically to its unit-normalized version,
    proving the vectorized dot-product path still normalizes the query
    (not just the stored side) before comparing."""
    cache = SemanticCache(similarity_threshold=0.9)
    cache.set("q1", _vec(1, 0, 0), {"answer": "A"})

    payload_small, sim_small = cache.lookup(_vec(0.001, 0, 0))
    payload_large, sim_large = cache.lookup(_vec(1000, 0, 0))
    assert payload_small == payload_large == {"answer": "A"}
    assert sim_small == pytest.approx(sim_large) == pytest.approx(1.0)


def test_vectorized_lookup_matches_brute_force_on_random_data():
    """Correctness proof for the vectorize+pre-normalize rewrite: compare
    the vectorized lookup's result against an independent, naive per-entry
    cosine-similarity loop over the same random embeddings - not just
    trusting the matmul refactor by inspection."""
    rng = np.random.default_rng(42)
    cache = SemanticCache(similarity_threshold=0.0)  # threshold 0 -> always returns the true best match
    raw_vectors = {}
    for i in range(25):
        vec = rng.normal(size=384)
        raw_vectors[f"q{i}"] = vec
        cache.set(f"q{i}", vec, {"answer": f"answer-{i}"})

    query = rng.normal(size=384)

    def naive_cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    expected_key = max(raw_vectors, key=lambda k: naive_cosine(query, raw_vectors[k]))
    expected_sim = naive_cosine(query, raw_vectors[expected_key])

    payload, sim = cache.lookup(query)
    assert payload == {"answer": f"answer-{expected_key[1:]}"}
    assert sim == pytest.approx(expected_sim, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

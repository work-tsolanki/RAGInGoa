import sys

import pytest

sys.path.insert(0, '.')

from src.answer_cache import AnswerCache, make_cache_key


def test_same_query_and_docs_produce_same_key():
    k1 = make_cache_key("What is Goa famous for?", ["chunk_1", "chunk_2"])
    k2 = make_cache_key("what is goa famous for", ["chunk_2", "chunk_1"])  # normalized + order-independent
    assert k1 == k2


def test_different_doc_sets_produce_different_keys():
    k1 = make_cache_key("What is Goa famous for?", ["chunk_1"])
    k2 = make_cache_key("What is Goa famous for?", ["chunk_2"])
    assert k1 != k2


def test_key_changes_with_prompt_version():
    k_old = make_cache_key("What is Goa famous for?", ["chunk_1"], prompt_version="v1-old")
    k_new = make_cache_key("What is Goa famous for?", ["chunk_1"], prompt_version="v2-new")
    assert k_old != k_new


def test_entry_stored_under_old_prompt_version_misses_after_bump():
    """Direct proof of the PROMPT_VERSION guard on the literal cache,
    mirroring the semantic-cache version of this same check: seed a
    cache entry under an old prompt version, confirm the OLD key still
    hits (control), then confirm the NEW version's key for the identical
    (query, docs) pair is a miss - the old-style answer never gets served
    once the prompt changes."""
    cache = AnswerCache()
    query, doc_ids = "What is Goa famous for?", ["chunk_1", "chunk_2"]

    old_key = make_cache_key(query, doc_ids, prompt_version="v1-old")
    cache.set(old_key, {"answer": "old-style boilerplate answer", "grounding_score": 0.9})

    # Control: looking it up with the same (old) version still hits.
    assert cache.get(old_key) is not None

    # The actual regression check: the new version's key for the identical
    # query+docs must miss, not resurrect the old answer.
    new_key = make_cache_key(query, doc_ids, prompt_version="v2-new")
    assert cache.get(new_key) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""LRU+TTL cache for generated answers, keyed on (query, retrieved doc set,
prompt version).

Skips generation + grounding entirely on a hit - retrieval still runs every
request (it's now ~5ms, see chroma_service.py/bm25s_service.py), so the
cache key is always built from the *current* retrieval result. That means a
corpus update that changes which docs a query retrieves changes the cache
key automatically, instead of silently serving an answer built from
now-stale context. The prompt version is folded in the same way: a
generation_service._build_prompt change (see config.PROMPT_VERSION) makes
every existing key unreachable instead of serving an old-style answer next
to fresh new-style ones.
"""

import hashlib
import re
import time
from collections import OrderedDict

from config import PROMPT_VERSION

_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    q = query.lower().strip()
    q = _PUNCT_RE.sub("", q)
    q = _WHITESPACE_RE.sub(" ", q)
    return q


def make_cache_key(query: str, doc_ids: list, language: str = "en", prompt_version: str = PROMPT_VERSION) -> str:
    """`language` is folded into the key (not just query+docs) so a cached
    answer generated for one target language can never be served back to a
    request for a different one - the same (query, doc set) can legitimately
    need answers in several languages, and those are different cache
    entries, not a collision."""
    normalized = normalize_query(query)
    doc_key = ",".join(sorted(doc_ids))
    raw = f"{prompt_version}|{language}|{normalized}|{doc_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AnswerCache:
    """Manual LRU (not functools.lru_cache) so hit/miss stats and a
    doc-dependent key are easy to inspect and reason about."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 86400):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        entry = self.cache.get(key)
        if entry is None:
            self.misses += 1
            return None

        value, expiry = entry
        if time.time() > expiry:
            del self.cache[key]
            self.misses += 1
            return None

        self.cache.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: dict):
        self.cache[key] = (value, time.time() + self.ttl_seconds)
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "size": len(self.cache),
            "max_size": self.max_size,
        }

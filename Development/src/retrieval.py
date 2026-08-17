from typing import List, Dict
from config import DEBUG
from src.latency_tracker import track_latency

def normalize_scores(results: List[Dict], max_score: float = None) -> List[Dict]:
    """Normalize scores to [0, 1]."""
    if not results:
        return []

    if max_score is None:
        max_score = max([r["score"] for r in results]) or 1

    return [
        {
            **r,
            "score": r["score"] / max_score if max_score > 0 else 0
        }
        for r in results
    ]

def _extract_language(result: Dict) -> str:
    """Dense (Chroma) results carry language under metadata; sparse (bm25s)
    results carry it top-level. Without this, merge_and_rank silently
    dropped language for any doc that only matched on the sparse side."""
    return result.get("language") or result.get("metadata", {}).get("language", "")


def _extract_field(result: Dict, field: str, default=None):
    """Same dual-shape handling as _extract_language, generalized: dense
    (Chroma) results carry extra fields under metadata; sparse (bm25s)
    results carry them top-level (see src/bm25s_service.py)."""
    if field in result:
        return result[field]
    return result.get("metadata", {}).get(field, default)


def dedupe_by_parent(candidates: List[Dict]) -> List[Dict]:
    """Collapses a whole passage and any of its own sub-chunks (see
    src/chunking/) down to whichever single one scored highest, when both
    appear in the same candidate set - otherwise a long passage could
    occupy multiple top-k slots with near-duplicate content under different
    chunk_ids. Active since scripts/add_chunking_strategies.py was run
    (154,744 fixed_overlap/semantic_boundary sub-chunks indexed for the
    24,462 passages over 100 tokens) - verified against a real query where
    a passage and its own sub-chunk both matched, collapsing correctly to
    the single highest-scoring one. For any passage that was never long
    enough to sub-chunk, parent_id still defaults to its own doc_id, so
    this is a true no-op for the ~97.6% of the corpus that's short passages."""
    best_by_parent = {}
    order = []
    for c in candidates:
        parent = c["parent_id"]
        if parent not in best_by_parent:
            order.append(parent)
            best_by_parent[parent] = c
        elif c["final_score"] > best_by_parent[parent]["final_score"]:
            best_by_parent[parent] = c
    return [best_by_parent[p] for p in order]


@track_latency("merge_results")
def merge_and_rank(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int = 5,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
    target_language: str = None,
) -> List[Dict]:
    """Merge dense + BM25 results with weighted fusion.

    target_language: if set, prefer docs in this language once ranked - the
    corpus (MSMARCO-XI) stores the same fact translated into many Indian
    languages as separate chunks, and multilingual embeddings correctly
    rank all of them as equally relevant. Left unfiltered, a single-language
    query can retrieve 4-5 different scripts for the same underlying fact,
    which confuses the LLM into answering in an unpredictable language. This
    keeps target_language matches first (still ranked by relevance among
    themselves) and only falls back to other languages to fill out top_k -
    never drops content solely because no same-language version exists.
    """

    merged = {}

    dense_results = normalize_scores(dense_results)
    for result in dense_results:
        doc_id = result["doc_id"]
        merged[doc_id] = {
            "doc_id": doc_id,
            "content": result.get("content", ""),
            "dense_score": result["score"],
            "bm25_score": 0.0,
            "language": _extract_language(result),
            "parent_id": _extract_field(result, "parent_id", doc_id),
            "chunking_strategy": _extract_field(result, "chunking_strategy"),
            "metadata": result.get("metadata", {})
        }

    bm25_results = normalize_scores(bm25_results)
    for result in bm25_results:
        doc_id = result["doc_id"]
        if doc_id in merged:
            merged[doc_id]["bm25_score"] = result["score"]
            if not merged[doc_id]["content"]:
                merged[doc_id]["content"] = result.get("content", "")
            if not merged[doc_id]["language"]:
                merged[doc_id]["language"] = _extract_language(result)
            if merged[doc_id]["chunking_strategy"] is None:
                merged[doc_id]["chunking_strategy"] = _extract_field(result, "chunking_strategy")
        else:
            merged[doc_id] = {
                "doc_id": doc_id,
                "content": result.get("content", ""),
                "dense_score": 0.0,
                "bm25_score": result["score"],
                "language": _extract_language(result),
                "parent_id": _extract_field(result, "parent_id", doc_id),
                "chunking_strategy": _extract_field(result, "chunking_strategy"),
                "metadata": result.get("metadata", {})
            }

    for doc_id in merged:
        merged[doc_id]["final_score"] = (
            dense_weight * merged[doc_id]["dense_score"] +
            bm25_weight * merged[doc_id]["bm25_score"]
        )

    candidates = sorted(
        merged.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )
    candidates = dedupe_by_parent(candidates)

    if target_language:
        same_lang = [d for d in candidates if d["language"] == target_language]
        other_lang = [d for d in candidates if d["language"] != target_language]
        ranked = (same_lang + other_lang)[:top_k]
    else:
        ranked = candidates[:top_k]

    if DEBUG:
        print(f"[merge_and_rank] Merged {len(merged)} docs -> top-{len(ranked)}")

    return ranked


if __name__ == "__main__":
    dense_results = [
        {"doc_id": "doc_001", "score": 0.95, "content": "Aadhaar is...", "metadata": {}},
        {"doc_id": "doc_002", "score": 0.80, "content": "आधार है...", "metadata": {}},
    ]

    bm25_results = [
        {"doc_id": "doc_001", "score": 8.5, "content": "Aadhaar is...", "metadata": {}},
        {"doc_id": "doc_003", "score": 7.2, "content": "Apply for Aadhaar", "metadata": {}},
    ]

    merged = merge_and_rank(dense_results, bm25_results, top_k=3)

    print("\nMerged results:")
    for r in merged:
        print(f"  {r['doc_id']}: {r['final_score']:.2f}")

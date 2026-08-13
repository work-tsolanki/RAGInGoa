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

@track_latency("merge_results")
def merge_and_rank(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int = 5,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4
) -> List[Dict]:
    """Merge dense + BM25 results with weighted fusion."""

    merged = {}

    dense_results = normalize_scores(dense_results)
    for result in dense_results:
        doc_id = result["doc_id"]
        merged[doc_id] = {
            "doc_id": doc_id,
            "content": result.get("content", ""),
            "dense_score": result["score"],
            "bm25_score": 0.0,
            "metadata": result.get("metadata", {})
        }

    bm25_results = normalize_scores(bm25_results)
    for result in bm25_results:
        doc_id = result["doc_id"]
        if doc_id in merged:
            merged[doc_id]["bm25_score"] = result["score"]
            if not merged[doc_id]["content"]:
                merged[doc_id]["content"] = result.get("content", "")
        else:
            merged[doc_id] = {
                "doc_id": doc_id,
                "content": result.get("content", ""),
                "dense_score": 0.0,
                "bm25_score": result["score"],
                "metadata": result.get("metadata", {})
            }

    for doc_id in merged:
        merged[doc_id]["final_score"] = (
            dense_weight * merged[doc_id]["dense_score"] +
            bm25_weight * merged[doc_id]["bm25_score"]
        )

    ranked = sorted(
        merged.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )[:top_k]

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

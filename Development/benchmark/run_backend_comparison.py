"""
Formal multi-backend generation comparison harness.

Reuses the exact production retrieval/generation/guardrails pipeline (not a
reimplementation) so results reflect what the live server actually does.
Adding a new backend later means adding one line to BACKEND_METHODS - the
rest of the script is backend-agnostic.

Usage:
    python benchmark/run_backend_comparison.py
    python benchmark/run_backend_comparison.py --backends groq,local
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '.')

from src.embedding_service import EmbeddingService
from src.chroma_service import ChromaService
from src.bm25s_service import Bm25sService
from src.retrieval import merge_and_rank
from src.generation_service import GenerationService
from src.guardrails import Guardrails

QUERIES_PATH = Path(__file__).parent / "queries.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

# Maps a backend name (as used in --backends and in results.json) to the
# GenerationService method that streams from it. Add new backends here.
BACKEND_METHODS = {
    "groq": "_generate_groq_stream",
    "cerebras": "_generate_cerebras_stream",
    "local": "_generate_local_stream",
    "claude": "_generate_claude_stream",
}


def run_single(query: str, lang: str, backend_name: str, backend_fn,
                embedding_service, chroma_service, bm25_service, generation_service, guardrails) -> dict:
    t0 = time.perf_counter()

    query_embedding = embedding_service.embed_query(query)
    dense = chroma_service.query(query_embedding.tolist(), top_k=10)
    sparse = bm25_service.query(query, top_k=10)
    merged = merge_and_rank(dense, sparse, top_k=5, target_language=lang)
    context_docs = [d["content"] for d in merged]
    t_retrieve = time.perf_counter()

    prompt = generation_service._build_prompt(query, context_docs, language=lang)
    answer_parts = []
    for delta in backend_fn(prompt):
        answer_parts.append(delta)
    answer = "".join(answer_parts).strip()
    t_generate = time.perf_counter()

    grounding_score = guardrails.check_grounding(answer, context_docs)
    is_valid = guardrails.validate_answer(answer)
    t_grounding = time.perf_counter()

    return {
        "query": query,
        "lang": lang,
        "backend": backend_name,
        "answer": answer,
        "answer_tokens_approx": len(answer.split()),
        "retrieval_ms": round((t_retrieve - t0) * 1000, 1),
        "generation_ms": round((t_generate - t_retrieve) * 1000, 1),
        "grounding_ms": round((t_grounding - t_generate) * 1000, 1),
        "total_ms": round((t_grounding - t0) * 1000, 1),
        "grounding_score": round(grounding_score, 3),
        "is_valid": is_valid,
    }


def print_summary_table(results):
    by_backend = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_backend[r["backend"]].append(r)

    print(f"\n{'Backend':<10} {'Avg Total ms':<14} {'Avg Gen ms':<12} {'Avg Grounding':<15} {'Valid %':<10} {'N':<5}")
    for backend, rows in by_backend.items():
        avg_total = sum(r["total_ms"] for r in rows) / len(rows)
        avg_gen = sum(r["generation_ms"] for r in rows) / len(rows)
        avg_grounding = sum(r["grounding_score"] for r in rows) / len(rows)
        valid_pct = sum(r["is_valid"] for r in rows) / len(rows) * 100
        print(f"{backend:<10} {avg_total:<14.1f} {avg_gen:<12.1f} {avg_grounding:<15.3f} {valid_pct:<10.1f} {len(rows):<5}")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n{len(errors)} backend call(s) failed:")
        for e in errors:
            print(f"  {e['backend']} / {e['query']!r}: {e['error']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="groq,local",
                         help="Comma-separated backend names to test (default: groq,local - "
                              "the two currently in production). Must be keys in BACKEND_METHODS.")
    args = parser.parse_args()
    backend_names = [b.strip() for b in args.backends.split(",") if b.strip()]

    unknown = [b for b in backend_names if b not in BACKEND_METHODS]
    if unknown:
        raise ValueError(f"Unknown backend(s) {unknown}. Known: {list(BACKEND_METHODS)}")

    print("Loading services...", flush=True)
    embedding_service = EmbeddingService()
    chroma_service = ChromaService(collection_name="hhgoa_rag_full")
    bm25_service = Bm25sService(index_dir="bm25s_index_full")
    generation_service = GenerationService(use_local=("local" in backend_names))
    guardrails = Guardrails()

    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Running {len(queries)} queries x {len(backend_names)} backends "
          f"({backend_names})...", flush=True)

    results = []
    for q in queries:
        for name in backend_names:
            backend_fn = getattr(generation_service, BACKEND_METHODS[name])
            try:
                result = run_single(
                    q["query"], q["lang"], name, backend_fn,
                    embedding_service, chroma_service, bm25_service, generation_service, guardrails,
                )
                results.append(result)
                print(f"  {name:<8} {q['lang']:<3} {q['query'][:40]:<42} "
                      f"{result['total_ms']:>7.0f}ms  grounding={result['grounding_score']:.3f}", flush=True)
            except Exception as e:
                results.append({"query": q["query"], "lang": q["lang"], "backend": name, "error": str(e)})
                print(f"  {name:<8} {q['lang']:<3} {q['query'][:40]:<42} FAILED: {e}", flush=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_summary_table(results)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

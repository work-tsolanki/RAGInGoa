"""
Runs a real query batch (30-50 queries, mixed en/hi/gu, spanning many real
corpus topics observed during this session - not the same 5-7 queries used
throughout development) through the LIVE ws://.../ws/query pipeline, then
pulls the resulting P50/P70/P100 percentiles from GET /metrics.

This exercises the actual production path (not benchmark/run_backend_
comparison.py, which calls generation_service/guardrails directly and
bypasses both the WS pipeline and the grounding gate), and now that
GenerationService.stream_generate() records into the "generation" latency
bucket (see src/generation_service.py), /metrics' percentiles reflect real
WS-pipeline traffic, not the single REST-only sample the Aug 2026 spec-
compliance audit found.

Paced (PACE_SECONDS between requests) to stay well under Groq's free-tier
rate limit - a rapid-fire batch was confirmed this session to cause
multi-second stalls that have nothing to do with system performance.
"""

import asyncio
import json
import sys
import time

sys.path.insert(0, '.')

import httpx
import websockets

from config import RAG_API_KEY

WS_URL = "ws://127.0.0.1:8000/ws/query"
METRICS_URL = "http://127.0.0.1:8000/metrics"
QUERIES_PATH = "benchmark/percentile_batch_queries.json"
RESULTS_PATH = "benchmark/percentile_batch_results.json"
PACE_SECONDS = 10.0


async def run_batch(queries: list) -> list:
    results = []
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"type": "auth", "api_key": RAG_API_KEY}))
        await ws.recv()

        for i, q in enumerate(queries):
            t0 = time.time()
            await ws.send(json.dumps({"query_text": q["query"], "language": q["lang"]}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                if msg.get("stage") == "final":
                    wall_ms = (time.time() - t0) * 1000
                    result = {
                        "query": q["query"], "lang": q["lang"],
                        "confidence": msg["confidence"], "hedged": msg.get("hedged", False),
                        "cache_hit": msg["cache_hit"], "backend": msg.get("backend"),
                        "total_ms": msg["latency_breakdown"]["total"],
                        "wall_ms": round(wall_ms, 1),
                    }
                    results.append(result)
                    print(f"[{i+1}/{len(queries)}] {q['lang']} hedged={result['hedged']!s:5s} "
                          f"conf={result['confidence']:.3f} total_ms={result['total_ms']:.0f} "
                          f"| {q['query'][:45]}", flush=True)
                    break
                elif msg.get("stage") == "error":
                    results.append({
                        "query": q["query"], "lang": q["lang"], "error": msg.get("message"),
                    })
                    print(f"[{i+1}/{len(queries)}] {q['lang']} ERROR: {msg.get('message')} "
                          f"| {q['query'][:45]}", flush=True)
                    break
            if i < len(queries) - 1:
                await asyncio.sleep(PACE_SECONDS)
    return results


def main():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Running {len(queries)} queries through the live WS pipeline "
          f"({PACE_SECONDS}s pacing between requests)...", flush=True)

    results = asyncio.run(run_batch(queries))

    errors = [r for r in results if "error" in r]
    ok_results = [r for r in results if "error" not in r]
    hedged = [r for r in ok_results if r["hedged"]]
    cache_hits = [r for r in ok_results if r["cache_hit"]]

    print(f"\n{len(ok_results)}/{len(results)} completed, {len(errors)} errors, "
          f"{len(hedged)} hedged ({len(hedged)/len(results)*100:.1f}%), "
          f"{len(cache_hits)} cache hits ({len(cache_hits)/len(results)*100:.1f}%)", flush=True)

    print("\nFetching /metrics for P50/P70/P100...", flush=True)
    metrics = httpx.get(METRICS_URL).json()

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "queries_run": len(results),
            "errors": len(errors),
            "hedged_count": len(hedged),
            "hedge_rate": len(hedged) / len(results) if results else 0,
            "cache_hit_count": len(cache_hits),
            "cache_hit_rate": len(cache_hits) / len(results) if results else 0,
            "per_query_results": results,
            "latency_metrics_p50_p70_p100": metrics["latency_metrics"],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n=== P50 / P70 / P100 (ms), per pipeline stage ===")
    for stage, stats in metrics["latency_metrics"].items():
        print(f"  {stage:<16} n={stats['count']:<4} "
              f"P50={stats['p50']:>8.1f}  P70={stats['p70']:>8.1f}  P100={stats['p100']:>8.1f}  "
              f"mean={stats['mean']:>8.1f}")

    print(f"\nFull results + percentiles written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

# Task: Full-path latency audit — find the generation latency regression

## Context

Generation latency has increased from the previously-measured ~175-270ms (Groq) baseline. Several changes landed recently that are all plausible causes: prompt template rewrite (natural-tone instructions + few-shot examples), `PROMPT_VERSION` cache-key versioning, semantic cache implementation/vectorization. This is not a "add more optimizations" task — it's a **regression hunt**: instrument every stage, on every request, and find exactly which stage's timing changed versus the last known-good baseline.

Do not guess or optimize blind. Do not touch retrieval, guardrails, or backend fallback logic during this audit — this is measurement only until the cause is confirmed.

## Step 1: Instrument every stage boundary, not just the ones you already track

Current instrumentation likely only wraps `retrieval_ms` / `generation_ms` / `grounding_ms` at a coarse level. Add a timer at every function boundary in the actual request path, including ones you haven't been isolating separately:

```python
import time
import json
import logging

log = logging.getLogger("latency_audit")

class StageTimer:
    def __init__(self):
        self.marks = {}
        self.t_start = time.perf_counter()
        self._last = self.t_start

    def mark(self, stage_name: str):
        now = time.perf_counter()
        self.marks[stage_name] = round((now - self._last) * 1000, 2)
        self._last = now

    def total(self):
        return round((time.perf_counter() - self.t_start) * 1000, 2)

    def log_summary(self, query: str, backend: str, cache_result: str):
        log.info(json.dumps({
            "query": query,
            "backend": backend,
            "cache_result": cache_result,  # "semantic_hit" | "literal_hit" | "miss"
            "stages_ms": self.marks,
            "total_ms": self.total(),
        }, ensure_ascii=False))
```

Wrap **every** stage individually, in the actual request handler — not just the big three:

```python
timer = StageTimer()

query_embedding = embed_query(query_text)
timer.mark("embed_query")

semantic_result, sim_score = semantic_cache.lookup(query_embedding)
timer.mark("semantic_cache_lookup")

if semantic_result is None:
    dense_r, sparse_r = await asyncio.gather(dense_search(query_embedding), sparse_search(query_text))
    timer.mark("hybrid_retrieval")

    fused = reciprocal_rank_fusion(dense_r, sparse_r, top_k=5)
    timer.mark("fusion")

    literal_key = make_cache_key(query_text, doc_ids, prompt_version=config.PROMPT_VERSION)
    literal_cached = answer_cache.get(literal_key)
    timer.mark("literal_cache_lookup")

    if literal_cached is None:
        prompt = build_prompt(query_text, fused)
        timer.mark("prompt_construction")

        answer = ""
        for delta in generate_with_fallback(prompt):
            answer += delta["delta"]
        timer.mark("generation")

        grounding_score = check_grounding(answer, fused)
        timer.mark("grounding_check")

        validated = validate_answer(answer, grounding_score)
        timer.mark("validation")

        if validated["is_valid"]:
            answer_cache.set(literal_key, result)
            semantic_cache.set(literal_key, query_embedding, result)
        timer.mark("cache_write")

timer.log_summary(query_text, backend_used, cache_result_type)
```

This is the critical difference from your existing benchmark harness: it instruments **every** boundary, including ones that were previously invisible (`embed_query`, `semantic_cache_lookup`, `prompt_construction`, `cache_write`), not just the three coarse stages you've been comparing against so far. The regression could be hiding in any of these, not necessarily in `generation_ms` itself.

## Step 2: Run against the smallest possible queries first

"Smallest and every query" from your ask means specifically: don't only test your normal benchmark set (which includes few-shot-sized prompts and moderate-length answers). Add genuinely trivial cases to isolate fixed overhead from variable, per-token cost:

```json
[
  {"query": "hi", "note": "minimal query, tests fixed overhead floor"},
  {"query": "GST", "note": "single-word query, minimal embedding/retrieval work"},
  {"query": "What is GST?", "note": "short, previously-fast query type"}
]
```

If even the trivial "hi" case shows elevated latency in a specific stage, that stage has a fixed-cost regression (something now runs unconditionally on every request, regardless of query complexity) rather than a scaling regression (something that only gets slow with more retrieved content or longer prompts). This distinction narrows the search significantly — check this before running your full benchmark set.

## Step 3: Compare directly against the last known-good baseline

You have this already — `benchmark/results.json` from before the prompt/cache changes. Don't eyeball it; diff programmatically:

```python
import json

def compare_baselines(old_path: str, new_path: str):
    with open(old_path) as f:
        old = {r["query"]: r for r in json.load(f) if "error" not in r}
    with open(new_path) as f:
        new = {r["query"]: r for r in json.load(f) if "error" not in r}

    for query in old:
        if query not in new:
            continue
        print(f"\n{query}")
        for stage in set(list(old[query].get("stages_ms", {}).keys()) +
                          list(new[query].get("stages_ms", {}).keys())):
            old_ms = old[query].get("stages_ms", {}).get(stage, "N/A")
            new_ms = new[query].get("stages_ms", {}).get(stage, "N/A")
            flag = ""
            if isinstance(old_ms, (int, float)) and isinstance(new_ms, (int, float)):
                delta = new_ms - old_ms
                if delta > 20:  # threshold: flag anything +20ms or more
                    flag = f"  <-- +{delta:.1f}ms"
            print(f"  {stage:<25} old={old_ms}  new={new_ms}{flag}")
```

This tells you, per query, per stage, exactly what changed — not a vague "generation got slower" impression.

## Step 4: Specific hypotheses to check first, given what changed recently

Check these in order — they're the most likely culprits based on the actual recent changes, before assuming it's something else entirely:

1. **`semantic_cache_lookup` timing** — was the vectorized numpy rewrite from the previous step actually applied, or is it still running the original per-entry Python loop? If the vectorization wasn't deployed yet, this stage alone could be adding real overhead now that the cache has entries in it (it was near-empty during earlier benchmarks).
2. **`prompt_construction` / few-shot token count** — confirm the few-shot examples actually pulled from your real corpus (per the earlier catch) rather than accidentally left as placeholder text, and check the actual token count added to every prompt. If this is larger than expected, it inflates `generation_ms` on every backend, worse on local fallback specifically — check whether Groq or local (or Claude) is currently serving requests, since a silent fallback to a slower backend would look exactly like "generation latency increased" without any prompt change being the cause at all.
3. **Cache hit rate drop from `PROMPT_VERSION` bump** — this was an intentional, expected side effect (old entries correctly miss under the new version), but if it wasn't anticipated in your latency expectations, a temporary flood of cache misses (each paying full retrieval+generation cost) while the cache repopulates under the new version could look like "generation got slower" when it's actually "more requests are now missing cache than before, temporarily." Check `cache_result` in your new logs — if hit rate cratered right after the version bump, this explains it and will self-resolve as the cache warms up again. Confirm this against timestamp of the version bump vs timestamp of the latency increase.
4. **Which backend actually served each request** — log `backend_used` on every request (should already be in your response payload per the Groq migration spec). If `generate_with_fallback` is silently falling through to local or Claude more often than before (e.g. Groq rate-limiting on the free/low tier, or a transient auth issue), that alone fully explains a latency jump with zero relation to any prompt or cache change. Check this before investigating anything else — it's the single fastest thing to rule in or out.

## Step 5: Isolate with a controlled A/B, not just historical comparison

Once you have a hypothesis from Step 4, confirm it directly rather than inferring from logs alone:

- **Backend hypothesis**: force `GENERATION_BACKEND_ORDER = ["groq"]` only (no fallback) temporarily, re-run the trivial + benchmark query sets, confirm whether latency returns to baseline. If yes, the regression is fallback-related, not prompt/cache-related.
- **Prompt-size hypothesis**: temporarily run with the old (pre-few-shot) prompt template against the same queries, same backend, compare `prompt_construction` + `generation` stage timings directly against the new template.
- **Cache-warmth hypothesis**: run the same query twice in a row under the new `PROMPT_VERSION` — first call is expected to miss (this is fine), but the second call should hit and be fast. If the second call is also slow, the regression isn't cache-related at all and you can rule this hypothesis out entirely.

## What to report back

For each stage in the audit, note: old baseline ms, new ms, delta, and which of the Step 4 hypotheses (if any) explains it. Do not apply a fix until the specific stage and specific cause are confirmed — a fix aimed at the wrong stage (e.g., further backend optimization when the real cause is a cache-hit-rate drop) wastes remaining time before the deadline without addressing the actual regression.

## What NOT to do during this audit

- Do not add new optimizations (further backend swaps, new caching layers, etc.) until the regression is identified — this compounds the number of variables and makes the next audit harder, not easier.
- Do not skip Step 4.4 (which backend served each request) — it's the fastest, cheapest check and rules out or confirms the single most likely external cause (backend fallback/rate-limiting) before you spend time auditing your own code.

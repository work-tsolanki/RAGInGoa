# Task: Add Mercury 2 (diffusion LLM) as a benchmarked generation backend, alongside other fast-inference angles worth testing

## Context

Current fallback chain: `Groq (llama-3.1-8b-instant, ~840 tok/s, autoregressive) → local llama.cpp → Claude API`. Generation latency is ~175-270ms, already near the floor for autoregressive decoding on any hardware. Mercury 2 uses a fundamentally different generation algorithm (diffusion/parallel token refinement instead of sequential decoding) and claims >1,000 tok/s. This is worth testing as a potential new primary backend — but it is unproven on this specific domain (Indic-language, retrieval-grounded, civic-FAQ RAG) and must clear an accuracy bar before replacing Groq, not just a speed bar.

This file covers: Mercury 2 integration, a proper multi-backend benchmark harness (you've been comparing backends informally on 3 queries each time — this formalizes it), and other angles worth a quick look while you're in this part of the stack.

## Part 1: Mercury 2 integration

### 1.1 Account setup (manual, not code)
1. Sign up at https://platform.inceptionlabs.ai
2. Generate an API key — new accounts get 100M free tokens, no payment info required.
3. Add `INCEPTION_API_KEY=<key>` to `.env` (not committed).

### 1.2 Config additions

In `config.py`:
```python
INCEPTION_API_KEY = os.environ.get("INCEPTION_API_KEY")
MERCURY_MODEL = "mercury-2"
MERCURY_MAX_TOKENS = 200       # match existing cap across all backends
MERCURY_TEMPERATURE = 0.3      # match existing setting
MERCURY_BASE_URL = "https://api.inceptionlabs.ai/v1"
```

### 1.3 Generation function

Mercury's API is OpenAI-compatible, so this can reuse the `openai` Python client rather than a new SDK. Add to `generation_service.py`:

```python
from openai import OpenAI

mercury_client = OpenAI(
    api_key=config.INCEPTION_API_KEY,
    base_url=config.MERCURY_BASE_URL,
) if config.INCEPTION_API_KEY else None

def generate_mercury(prompt: str, max_tokens: int = None, temperature: float = None):
    if mercury_client is None:
        raise RuntimeError("INCEPTION_API_KEY not configured")

    stream = mercury_client.chat.completions.create(
        model=config.MERCURY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens or config.MERCURY_MAX_TOKENS,
        temperature=temperature or config.MERCURY_TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

Use the exact same `_build_prompt` output as every other backend — instruction template, retrieved passages, language-match instruction, all identical. Do not write a Mercury-specific prompt.

**Note on Mercury 2's "tunable reasoning levels"**: the model supports adjustable reasoning depth. For this RAG use case (short factual synthesis from retrieved passages, not open-ended reasoning), test with reasoning set to its lowest/fastest tier first — a full-reasoning mode is very likely unnecessary overhead for "summarize these 5 passport-office passages into an answer" and would only add latency without adding value here. Check Inception's docs for the exact parameter name at implementation time, since this is a newer feature and naming may still be settling.

### 1.4 Add to fallback chain

Update `config.py`:
```python
GENERATION_BACKEND_ORDER = ["mercury", "groq", "local", "claude"]
```

The `generate_with_fallback` dispatcher from the Groq migration already handles this — just add `"mercury": generate_mercury` to its backend dict. No other changes needed there.

**Do not set this as the live default order yet.** Keep `["groq", "local", "claude"]` as the production default until Part 2's benchmark passes. Use an environment flag or a separate test endpoint to exercise the Mercury-first order during evaluation.

## Part 2: Formal multi-backend benchmark harness

You've compared backends informally throughout this project (3 manual queries each time). Build this once, properly, and reuse it for every future backend comparison — including whatever comes after Mercury.

### 2.1 Build a fixed benchmark query set

Create `benchmark/queries.json` — do this once, keep it stable across all future comparisons so results are comparable over time:

```json
[
  {"query": "How to apply for a passport", "lang": "en"},
  {"query": "What is a corporation?", "lang": "en"},
  {"query": "income tax filing deadline", "lang": "en"},
  {"query": "What are voter ID requirements", "lang": "en"},
  {"query": "How does GST registration work", "lang": "en"},
  {"query": "पासपोर्ट के लिए आवेदन कैसे करें", "lang": "hi"},
  {"query": "મતદાર ID માટે શું જરૂરી છે", "lang": "gu"}
]
```
Include at least the Hindi/Gujarati queries — this is the exact category where you previously caught the language-mismatch bug, and it's the most likely place a new backend (trained/tuned differently than Llama 3.1) could regress.

### 2.2 Harness script

Create `benchmark/run_backend_comparison.py`:

```python
import json
import time
import asyncio
from src.retrieval import hybrid_retrieve  # reuse existing retrieval pipeline
from src.generation_service import generate_with_fallback, generate_mercury, generate_groq, generate_local
from src.guardrails import check_grounding, validate_answer
from src.prompt_builder import build_prompt

BACKENDS = {
    "mercury": generate_mercury,
    "groq": generate_groq,
    "local": generate_local,
}

async def run_single(query: str, backend_fn, backend_name: str):
    t0 = time.perf_counter()
    fused_docs, doc_ids = await hybrid_retrieve(query)  # existing retrieval
    t_retrieve = time.perf_counter()

    prompt = build_prompt(query, fused_docs)
    answer = ""
    for delta in backend_fn(prompt):
        answer += delta
    t_generate = time.perf_counter()

    grounding_score = check_grounding(answer, fused_docs)
    validated = validate_answer(answer, grounding_score)
    t_grounding = time.perf_counter()

    return {
        "query": query,
        "backend": backend_name,
        "answer": validated["answer"],
        "answer_tokens_approx": len(answer.split()),
        "retrieval_ms": round((t_retrieve - t0) * 1000, 1),
        "generation_ms": round((t_generate - t_retrieve) * 1000, 1),
        "grounding_ms": round((t_grounding - t_generate) * 1000, 1),
        "total_ms": round((t_grounding - t0) * 1000, 1),
        "grounding_score": round(grounding_score, 3),
        "is_valid": validated["is_valid"],
    }

async def main():
    with open("benchmark/queries.json") as f:
        queries = json.load(f)

    results = []
    for q in queries:
        for name, fn in BACKENDS.items():
            try:
                result = await run_single(q["query"], fn, name)
                result["lang"] = q["lang"]
                results.append(result)
            except Exception as e:
                results.append({"query": q["query"], "backend": name, "error": str(e)})

    with open("benchmark/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_summary_table(results)

def print_summary_table(results):
    from collections import defaultdict
    by_backend = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_backend[r["backend"]].append(r)

    print(f"{'Backend':<10} {'Avg Total ms':<14} {'Avg Grounding':<15} {'Valid %':<10}")
    for backend, rows in by_backend.items():
        avg_ms = sum(r["total_ms"] for r in rows) / len(rows)
        avg_grounding = sum(r["grounding_score"] for r in rows) / len(rows)
        valid_pct = sum(r["is_valid"] for r in rows) / len(rows) * 100
        print(f"{backend:<10} {avg_ms:<14.1f} {avg_grounding:<15.3f} {valid_pct:<10.1f}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2.3 Manual review step (do not skip — this is the part that actually matters)

The script gives you speed and grounding-score numbers automatically. It cannot tell you if an answer is actually correct, coherent, or in the right language. After running it:

1. Open `benchmark/results.json`, read every Mercury answer next to its Groq/local counterpart for the same query.
2. Specifically check the Hindi/Gujarati rows for language-match correctness — this is where you've had a real bug before, so it's the highest-risk category for a new backend.
3. Flag anything that reads as fluent-but-wrong (a diffusion model "editing a full draft at once" can plausibly produce fluent, confident, incorrect text — the failure mode may look different from an autoregressive model's failure mode, so don't assume your existing intuitions about what a bad answer looks like transfer directly).

### 2.4 Decision rule

Only promote Mercury to primary (`GENERATION_BACKEND_ORDER = ["mercury", "groq", "local", "claude"]` in production config) if, across the full query set:
- Average `total_ms` is meaningfully lower than Groq's current numbers, AND
- Average `grounding_score` is comparable to or better than Groq's (not just "not terrible" — comparable), AND
- Manual review of the Hindi/Gujarati rows shows no language-match regressions, AND
- `is_valid` rate matches Groq's

If any of these fail, keep Groq primary and note Mercury as "benchmarked, not promoted" rather than discarding the integration — the code stays, just not in the live default path.

## Part 3: Other angles worth a quick look while benchmarking

These are lower-priority than Mercury 2 itself — do them only if time remains after Part 2, and only as additions to the same harness, not separate investigations.

### 3.1 Cerebras (autoregressive, but very fast hardware)
Add `generate_cerebras` the same way as `generate_groq` — same Llama-3.1-8B model, different silicon (~1,800 tok/s claimed vs Groq's ~840). Since it's the same model weights, expect answer quality to be nearly identical to Groq — this test is really just "is Cerebras's speed claim real for our prompt lengths and does their uptime hold up," not an accuracy question. Add to the same benchmark harness as a fourth backend.

### 3.2 Output token cap tuning, per backend
You already found (with k=5 retrieval) that most answers land well under 200 tokens. Since Mercury's per-step cost structure differs from autoregressive (T denoising steps, not per-token generation), the `max_tokens` knob may not map onto latency the same way it does for Groq/local. Once Mercury is integrated, test whether capping tighter (e.g. 100) changes its latency proportionally the way it does for autoregressive backends, or whether Mercury's latency is dominated by the fixed denoising-step count instead. This affects whether "shorten the prompt/output" is even a useful lever for Mercury specifically.

### 3.3 Do NOT pursue these right now
- **Custom-training or fine-tuning a diffusion model** — far outside your timeline.
- **Self-hosting a dLLM** (LLaDA, Dream) on your own GPU — as of the current research, native serving-framework support for dLLMs is still early/unstable; this is a research project, not a week-before-deadline task. Hosted Mercury 2 gets you the architecture's benefits without that risk.
- **Gemini Diffusion** — not open-source/generally API-accessible at the time of writing; skip unless that's changed.

## Testing checklist (full part 1 + 2)

- [ ] Mercury account created, free tokens confirmed available
- [ ] `generate_mercury` unit-tested standalone (mocked client) for correct streaming shape
- [ ] `generate_with_fallback` updated to include `"mercury"` in its backend dict
- [ ] Benchmark query set (`queries.json`) created, includes English + Hindi + Gujarati
- [ ] Harness runs cleanly across all three backends (mercury, groq, local), writes `results.json`
- [ ] Manual review of every Mercury answer completed, with specific attention to Hindi/Gujarati rows
- [ ] Decision made and documented: promote to primary, or keep as benchmarked-but-not-live
- [ ] If promoted: production `GENERATION_BACKEND_ORDER` updated, kill-switch test repeated (invalid API key → falls through to Groq) same as was done for the original Groq migration
- [ ] Answer cache and semantic cache still function correctly regardless of which backend produced the cached answer (cache logic is backend-agnostic — this should already hold, just confirm)

## What NOT to change

- Retrieval pipeline — untouched.
- `answer_cache.py` / semantic cache internals — untouched, backend-agnostic by design.
- `guardrails.py` — untouched; it treats all backend output identically.
- Prompt template (`_build_prompt`) — must stay identical across every backend under test. Any backend-specific prompt tweaking invalidates the comparison.

## Suggested order of work

1. Part 1 (Mercury integration) — mechanical, ~30-45 min.
2. Part 2.1–2.2 (harness + query set) — build once, ~1 hr, reusable forever.
3. Part 2.3–2.4 (manual review + decision) — the step that actually determines whether any of this ships. Do not skip or rush this in favor of the speed numbers alone.
4. Part 3 — only if time remains post-deadline-crunch; not required for the Mercury decision itself.

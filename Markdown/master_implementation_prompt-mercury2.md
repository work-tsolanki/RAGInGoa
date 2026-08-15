# Master Implementation Prompt: Voice RAG System — Generation Layer Finalization

## Project context

Voice-enabled RAG system, Task 2 submission for HH Goa 2026 (deadline: **August 22, 2026**). Dataset: AI4Bharat MSMARCO-XI (Indic-language civic/government FAQ content — passports, tax, voter ID, GST, etc.). Solo developer, terminal-agent-driven implementation.

You are implementing the remaining generation-layer work on an otherwise-finished, already-optimized system. Retrieval and caching are done and must not be touched. Your job is to finish the generation backend layer, formalize evaluation, and make a data-driven decision about final production configuration before the deadline.

## Current system state (verified, working, do not re-optimize)

- **Retrieval**: hybrid dense (ChromaDB HNSW) + sparse (bm25s) retrieval, RRF fusion, top_k=5, ~40ms total wall time (parallel dense+sparse). Confirmed via direct timing instrumentation. **Do not touch.**
- **Answer cache** (`src/answer_cache.py`): exact-match LRU, keyed on `normalize_query(query) + sorted(doc_ids)`, SHA-256, TTL 24h, max 1000 entries, hit/miss stats. Only caches answers that pass `guardrails.validate_answer()`. Wired into `/query` and WebSocket handler. Confirmed ~12ms on hit vs ~1550ms on miss. **Do not touch cache internals.**
- **Semantic cache** (`src/semantic_cache.py`, if already implemented per prior spec): fuzzy layer in front of the literal cache, cosine similarity on query embeddings, calibrated threshold. If not yet implemented, this is separate prior work — reference `semantic_cache_implementation.md` — not part of this prompt's scope unless explicitly resumed.
- **Generation backends currently wired**:
  - Groq (`llama-3.1-8b-instant`, ~840 tok/s, autoregressive) — **current production primary**
  - Local llama.cpp (`Llama-3.1-8B-Instruct` Q4_K_M, Flash Attention on, ~130 tok/s) — fallback #2
  - Claude API — fallback #3
  - Fallback chain implemented via `generate_with_fallback()` in `generation_service.py`, ordered by `config.GENERATION_BACKEND_ORDER`
- **Streaming**: tokens streamed to client as generated; full text buffered server-side for grounding + caching after stream completes.
- **Guardrails** (`guardrails.py`): `check_grounding` (cross-encoder mmarco-mMiniLMv2-L12-H384-v1), `validate_answer`. Currently `ANSWER_CACHE_MIN_GROUNDING = 0.0` (uncalibrated — real scores observed range 0.02-0.67 with no clean separation found yet). **This is an open item, see Priority 2 below.**
- **Prompt template** (`_build_prompt` / `prompt_builder.py`): includes a language-matching instruction (fix for a confirmed bug where English queries returned Gujarati/Sanskrit answers due to mixed-language retrieved passages). This instruction was added but never quantitatively re-verified.
- **Latency, current measured state**: retrieval ~40ms, generation ~175-270ms (Groq), grounding ~15-20ms, cache hit ~12-15ms. This is near the practical floor for autoregressive generation on hosted fast-inference hardware.

## What this prompt covers, in priority order

Work through these in order. Each has a decision gate — do not proceed past a gate without the evidence it requires. Given the deadline, treat Priority 1 as required, Priority 2 as required, Priority 3 as high-value-if-time-allows, Priority 4 as optional/stretch.

---

### Priority 1 (required): Formal multi-backend benchmark harness

Before adding any new backend, build the reusable evaluation harness — every backend decision from here forward should run through this, not ad hoc manual query tests.

1. Create `benchmark/queries.json`: a fixed, stable set of test queries — minimum 5 English + 2+ Indic-language (Hindi, Gujarati) queries drawn from real usage patterns (passport, tax, voter ID, GST, corporation topics). This set does not change between benchmark runs; that's what makes results comparable over time.
2. Create `benchmark/run_backend_comparison.py`: for each query × each registered backend, run retrieval → generation → grounding → validation, and record: retrieval_ms, generation_ms, grounding_ms, total_ms, grounding_score, is_valid, and the full answer text. Output to `benchmark/results.json`.
3. Print a summary table (avg total_ms, avg grounding_score, valid % per backend).
4. **Gate**: harness must run cleanly against the two currently-wired backends (Groq, local) before any new backend is added, to confirm the harness itself is correct.

Full script reference: `mercury2_and_alt_architectures.md`, Part 2.

---

### Priority 2 (required): Calibrate the grounding threshold

This has been deferred twice already and directly affects both the answer cache and any future backend-promotion decision (you cannot compare backends on grounding quality against an uncalibrated, effectively-disabled threshold).

1. Using the benchmark harness output plus any additional real logged answers, assemble 30-50 labeled (grounded-correct vs not) query/answer pairs.
2. Plot grounding score against label. Identify where the distributions actually separate — if they don't separate cleanly, document that finding; it's real information about the cross-encoder's discriminative power on this domain, not a failure of the exercise.
3. Set `config.ANSWER_CACHE_MIN_GROUNDING` from this data.
4. **Gate**: do not promote any new generation backend to production primary (Priority 3) using an uncalibrated threshold — you need a real bar to compare against.

---

### Priority 3 (high value): Integrate and evaluate Mercury 2 (diffusion LLM)

Mercury 2 (Inception Labs) uses parallel/diffusion-based generation instead of autoregressive decoding, claiming >1,000 tok/s — architecturally different from every backend tested so far, not just faster hardware running the same approach.

1. Sign up at platform.inceptionlabs.ai, get API key (100M free tokens, no payment info required), add `INCEPTION_API_KEY` to `.env`.
2. Implement `generate_mercury()` in `generation_service.py` — OpenAI-compatible client, same streaming interface as `generate_groq`/`generate_local`, same `_build_prompt` output (no backend-specific prompt changes).
3. Add `"mercury"` to the `generate_with_fallback` backend dict. Do **not** change production `GENERATION_BACKEND_ORDER` yet.
4. Run the Priority 1 harness with Mercury included as a third backend.
5. **Manual review is mandatory, not optional**: read every Mercury answer next to Groq's answer for the same query, with specific attention to the Hindi/Gujarati rows — this is the exact category where a real language-mismatch bug was previously found, and a diffusion model's failure mode (fluent-but-wrong, since it edits a full draft rather than generating token-by-token) may not resemble failure modes you've already learned to spot.
6. **Promotion gate** — only set `GENERATION_BACKEND_ORDER = ["mercury", "groq", "local", "claude"]` in production if, across the full benchmark set: average total_ms is meaningfully lower than Groq, average grounding_score is comparable or better (using the Priority 2 calibrated threshold, not the old 0.0), manual review shows no language-match regressions, and is_valid rate matches Groq. If any fail, keep Groq primary, keep Mercury integrated-but-not-live, document why.

Full spec: `mercury2_and_alt_architectures.md`, Part 1 + 2.

---

### Priority 4 (optional, only if time remains): Secondary angles

- **Cerebras**: same Llama-3.1-8B model as Groq, different silicon (~1,800 tok/s claimed). Since it's the same weights, this is a speed/reliability test, not an accuracy test — add as a fourth harness backend if time allows.
- **Per-backend output token cap tuning**: Mercury's diffusion step-count cost structure may not respond to `max_tokens` the same way autoregressive backends do — worth a quick check once Mercury is integrated, not before.
- **Explicitly out of scope for this deadline**: self-hosting any open dLLM (LLaDA, Dream) — serving-framework support is still immature; fine-tuning any model; Gemini Diffusion (not generally API-accessible at time of writing).

---

## Non-negotiable constraints across all priorities

- **Never modify**: retrieval pipeline, answer cache / semantic cache internal logic, `guardrails.py` internals, the core prompt template structure (only the language-match instruction has been deliberately added — no further backend-specific prompt forking).
- **Every new backend** must plug into the existing `generate_with_fallback` pattern — same function signature, same streaming shape, same error-handling contract (raise before yielding anything → clean fallthrough; fail mid-stream → do not silently retry on a different backend for that request).
- **Every backend-promotion decision** requires the harness + manual review, not a handful of ad hoc queries. This has been the working pattern throughout the project (see the k=5 vs k=10 retrieval comparison, the Groq benchmark, the RRF fusion validation) — continue it.
- **Document every "tested, not promoted" outcome** (config left as-is, reasoning noted) as clearly as every "tested, promoted" outcome — both are valid results of a benchmark, and future-you needs to know what was already ruled out and why.

## Final rollout checklist (before Aug 22 submission)

- [ ] Benchmark harness built and validated against existing backends
- [ ] Grounding threshold calibrated from real labeled data, config updated
- [ ] Mercury 2 integrated, benchmarked, manually reviewed
- [ ] Backend promotion decision made and documented either way
- [ ] If any production config changed: kill-switch test repeated (invalid API key on whichever backend is now primary → clean fallthrough to next backend, no hang, no user-facing failure)
- [ ] Answer cache + semantic cache confirmed working correctly regardless of which backend produced the underlying answer
- [ ] Full end-to-end smoke test on the 3 original benchmark queries plus the Hindi/Gujarati set, from cold start, on the final production configuration

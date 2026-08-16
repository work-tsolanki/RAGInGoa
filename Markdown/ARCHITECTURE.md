# System Architecture & Design

This document describes what is actually implemented and deployed, not the
original pre-implementation plan. Earlier drafts of this file described
Pinecone, Whoosh, and a fixed local-Llama/Claude elapsed-time switch - none
of that shipped. This version was rewritten against the real code and the
live deployment (`https://ragingoa.fly.dev`) as of the Aug 2026 production
hardening pass, per the spec-compliance audit's finding that this document
had drifted from the implementation once already.

---

## 🏗️ High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                         Client (dashboard / API caller)                │
│           Text query, or mic recording via the dashboard's WS UI       │
└───────────────────────────┬─────────────────────────────────────────┬─┘
        REST: POST /query, /demo/query, /query_audio   │   WS: /ws/query, /ws/demo
        (X-API-Key header)                              │   (auth frame, or none on /ws/demo)
                            │                            │
                            ▼                            ▼
┌───────────────────────────────────────────────────────────────────────┐
│                          FastAPI app (main_app.py)                     │
│                                                                          │
│  0. Access gate                                                         │
│     /query, /ws/query, /query_audio  -> X-API-Key / WS auth frame       │
│     /demo/query, /ws/demo            -> per-IP rate limit instead       │
│                                          (Fly-Client-IP, not spoofable  │
│                                          X-Forwarded-For), same pipeline│
│                                                                          │
│  1. (WS voice path only) STT - Sarvam saaras:v3-realtime, streamed      │
│     transcript as the recording happens; batch STT for one-shot audio  │
│                                                                          │
│  2. Unsafe-input check (regex, obvious-intent phrases) - reject before │
│     spending an embedding call on it                                   │
│                                                                          │
│  3. Embed query - sentence-transformers/paraphrase-multilingual-       │
│     MiniLM-L12-v2, 384-dim, shared across embedding/off-topic gate     │
│                                                                          │
│  4. Off-topic check - cosine similarity vs a calibrated reference-query │
│     set (scripts/calibrate_off_topic_threshold.py); reject before      │
│     spending a cache lookup or retrieval on it                         │
│                                                                          │
│  5. Semantic cache lookup (embedding similarity >= 0.92) - a hit skips  │
│     retrieval AND generation entirely, not just generation             │
│                                                                          │
│  6. Parallel retrieval                                                 │
│       Dense: Chroma (persistent, HNSW)     Sparse: bm25s (sparse matmul │
│       858,768-doc collection, volume-        over a memory-mapped index,│
│       mounted in production                  same 858,768-doc corpus)  │
│                                                                          │
│  7. Merge & rank (src/retrieval.py::merge_and_rank) - weighted fusion  │
│     of dense + sparse scores, then dedupe_by_parent collapses a whole  │
│     passage and its own sub-chunks down to whichever scored highest    │
│                                                                          │
│  8. Literal answer cache lookup (exact query + retrieved-doc-set key)  │
│                                                                          │
│  9. Generation - ordered fallback chain (config.GENERATION_BACKEND_    │
│     ORDER): Groq (llama-3.1-8b-instant) -> local llama.cpp (dev only,  │
│     see Decision 2) -> Claude (claude-3-haiku-20240307). Production    │
│     overrides this to Groq -> Claude only (no GPU on Fly - see         │
│     Decision 2 and Deployment below). temperature=0.1 on every backend.│
│                                                                          │
│ 10. Grounding check - multilingual cross-encoder (cross-encoder/       │
│     mmarco-mMiniLMv2-L12-H384-v1) scores the generated answer against  │
│     each retrieved passage; below ANSWER_CACHE_MIN_GROUNDING (0.7) the │
│     returned answer is replaced with a language-aware hedge message -  │
│     this gates what's actually returned, not just what gets cached     │
│                                                                          │
│ 11. Format & return - REST: single JSON response. WS: one JSON event   │
│     per stage (embedding/retrieval/generation/grounding start+done)    │
│     followed by a final event, so the dashboard shows live progress    │
└───────────────────────────┬─────────────────────────────────────────┬─┘
                    REST: JSON response                    WS: stream of stage events
                            ▼                                          ▼
                     Client receives                          Client receives live
                     answer + retrieved docs                  progress + final answer
                     + confidence + latency                   + confidence + latency
                     breakdown                                 breakdown
```

---

## 🔌 Real Service Modules

| Module | Role |
|---|---|
| `src/embedding_service.py` | Wraps the sentence-transformers embedding model. |
| `src/chroma_service.py` | Dense retrieval - persistent Chroma collection (`hhgoa_rag_full`), path configurable via `CHROMA_PERSIST_DIR` (points at the mounted volume in production). |
| `src/bm25s_service.py` | Sparse retrieval - `bm25s`, memory-mapped sparse-matmul index, path via `BM25S_INDEX_DIR`. Replaced Whoosh (see Decision 4) after Whoosh was found returning zero results on most natural-language queries. |
| `src/retrieval.py` | `merge_and_rank` (weighted dense+sparse fusion) and `dedupe_by_parent` (collapses a passage and its own sub-chunks). |
| `src/chunking/fixed_overlap.py`, `src/chunking/semantic.py` | The two additive sub-chunking strategies - see Decision 3. |
| `src/generation_service.py` | `generate()` (REST, blocking) and `stream_generate()` (WS, token-by-token) over the same ordered backend fallback chain. Both record into the same `latency_tracker` "generation" bucket. |
| `src/guardrails.py` | `check_grounding` (cross-encoder), `validate_answer` (refusal-phrase/empty-answer detection), `check_off_topic` / `check_unsafe` (the pre-retrieval gate - see below), plus the language-aware hedge/decline response builders. |
| `src/answer_cache.py` | Literal (exact-match) answer cache, keyed on `(query, retrieved_doc_ids)`. |
| `src/semantic_cache.py` | Fuzzy cache keyed on embedding similarity - catches reworded repeats the literal cache would miss. |
| `src/rate_limiter.py` | In-memory per-IP sliding-window limiter, backing the `/demo/query` and `/ws/demo` access gate. |
| `src/latency_tracker.py` | Records per-stage latency samples; backs `GET /metrics` (P50/P70/P100 per stage). |
| `src/stt_service.py` | Sarvam `saaras:v3-realtime` for streamed mic input, plus a batch path for one-shot audio uploads. |

---

## 🛂 The off-topic / unsafe-input gate

Added during production hardening after the spec audit flagged it as a
named-but-missing requirement ("handling for off-topic queries, unsafe/
inappropriate inputs"). Runs in `main_app.py`, before the semantic/literal
cache lookup - no retrieval or generation cost is spent on a query that was
never going to be answered.

- **Unsafe check** (`check_unsafe`): a small set of obvious-intent regex
  patterns (self-harm, weapons, illegal-drug synthesis, CSAM). Pure text
  match, no model call - runs first, before embedding.
- **Off-topic check** (`check_off_topic`): the corpus (MSMARCO-XI) is
  open-domain general-knowledge QA, not a narrow topic - there's no bounded
  "list of in-scope subjects" to enumerate. Instead, the query's embedding
  is compared (cosine similarity) against a fixed reference set of ~35
  example queries spanning the corpus's actual observed breadth
  (`_OFF_TOPIC_REFERENCE_QUERIES` in `src/guardrails.py`), including a few
  Hindi/Gujarati anchors since the multilingual embedding space isn't
  perfectly language-symmetric (a real, measured gap - the same phenomenon
  behind the cross-lingual hedge-rate finding in the Aug 2026 benchmark).
  `OFF_TOPIC_SIMILARITY_THRESHOLD = 0.499`, calibrated (not chosen by eye)
  via `scripts/calibrate_off_topic_threshold.py` against this project's own
  real benchmark queries (must stay above threshold) vs. deliberately
  off-topic ones - creative writing, pure computation, casual chat, meta
  questions about the assistant - which must stay below it.

Both checks apply identically on the authenticated (`/query`, `/ws/query`)
and public demo (`/demo/query`, `/ws/demo`) paths.

---

## 🌐 Access paths

| Path | Auth | Notes |
|---|---|---|
| `POST /query` | `X-API-Key` header | Full pipeline, single JSON response. |
| `WS /ws/query` | First message `{"type":"auth","api_key":...}` | Streaming stage events; also carries the voice flow (`audio_stream_start`/`audio_chunk`/`audio_stream_end`, or one-shot `audio_query`). |
| `POST /demo/query` | None - rate-limited | Identical pipeline to `/query`, gated by `verify_demo_rate_limit` instead of a key. |
| `WS /ws/demo` | None - rate-limited | Identical to `/ws/query` (text and voice both work), gated per-query-submission by the same rate limiter. This is what the public dashboard connects to by default, so a judge can use it with zero setup. |
| `POST /query_audio` | `X-API-Key` header | One-shot audio upload, REST response. |
| `GET /health` | None | Liveness/readiness. |
| `GET /metrics` | None | Latency percentiles per pipeline stage. |
| `GET /dashboard` | None | Serves the static dashboard shell - never server-injects the API key (see Security below). |

The dashboard's WebSocket connection can't set custom headers, so the
authenticated WS path uses a first-message auth frame rather than a
query-string key (which would land in access logs and browser history).
The demo path skips this entirely - no key is ever requested, held, or
transmitted by the public dashboard.

Rate limit on the demo path: 10 requests per 60 seconds per IP
(`DEMO_RATE_LIMIT_MAX_REQUESTS`/`_WINDOW_SECONDS` in `main_app.py`),
keyed on Fly's `Fly-Client-IP` header (set by Fly's edge, not
client-controlled) rather than the spoofable `X-Forwarded-For`. For a WS
connection, the limit applies per query submitted, not per raw WS message -
a voice recording's many small `audio_chunk` messages don't each count
against the budget.

---

## 🧠 Design Decisions

### Decision 1: Hybrid Retrieval (Dense + BM25), on Chroma + bm25s

**Options considered**: dense-only (fast, misses keywords), BM25-only
(keywords, misses semantic meaning), hybrid (chosen), ColBERT (too slow at
this scale).

Implemented as Chroma (dense, HNSW) + `bm25s` (sparse, memory-mapped)
running in parallel per query, fused by `merge_and_rank`. (Not Pinecone/
Whoosh, which an earlier draft of this document described but were never
actually integrated for the shipped system - see Decision 4 for the
Whoosh→bm25s switch specifically.)

---

### Decision 2: Ordered generation fallback (Groq -> local -> Claude), local excluded in production

**Options**: always-local (fast, lower quality), always-API (best quality,
slower/costlier), an elapsed-time-triggered switch between them (the
original plan), an ordered fallback chain (chosen).

`GENERATION_BACKEND_ORDER = ["groq", "local", "claude"]` - each backend is
tried in order; a failure or empty stream falls through to the next, not
an elapsed-time budget check. Groq (`llama-3.1-8b-instant`) is fast enough
in practice that the elapsed-time-switch design in the original plan was
unnecessary. `temperature=0.1` on every backend (lowered from an earlier
0.3 during production hardening - the live deployment's own benchmark
caught the same query scoring grounded on one run and hedged on a repeat,
phrasing variance from temperature feeding into the grounding-score
jitter, not a retrieval difference).

**Local llama.cpp is excluded in production** (`GENERATION_BACKEND_ORDER_
OVERRIDE=groq,claude` set in `fly.toml`'s `[env]`): Fly's standard machines
have no GPU, so local generation would only ever fail there. No code was
deleted - `GenerationService` only imports `llama_cpp` when the GGUF model
file exists on disk, and that file is deliberately excluded from the
production Docker image (`.dockerignore`), so the import path is simply
never reached.

---

### Decision 3: Chunking strategy - what's actually implemented

**Base layer (the vast majority of the corpus):** `scripts/download_dataset.py`
uses MSMARCO-XI's own pre-segmented passages directly as retrieval units -
one dataset passage = one whole-passage chunk (`data/msmarco-xi/chunks.jsonl`,
743,739 rows). No splitting/windowing is applied to these; only `chunk_id`,
`content`, `language`, `source`, `query_id` metadata.

**Additive second and third layer (`src/chunking/`, `scripts/add_chunking_
strategies.py`):** passages exceeding `LENGTH_THRESHOLD_TOKENS` (100) get
further split two ways, indexed *alongside* the originals (never replacing
them):
- **Fixed-overlap** (`src/chunking/fixed_overlap.py`): sliding word-count
  windows (`window_tokens=100`, `overlap_tokens=20`).
- **Semantic-boundary** (`src/chunking/semantic.py`): splits between
  sentences where embedding cosine similarity drops below a threshold,
  rather than at a fixed size.

Every chunk (base or sub-chunk) stores:
```json
{
  "doc_id": "chunk_id",
  "content": "chunk text",
  "language": "en|hi|ta|te|etc",
  "parent_id": "the whole-passage chunk_id this came from (self, for base chunks)",
  "chunking_strategy": "fixed_overlap | semantic_boundary | null (base passage)"
}
```
`parent_id` is used in `src/retrieval.py::dedupe_by_parent` to collapse a
whole passage and its own sub-chunk down to whichever scored higher, if
both land in the same top-k set. `chunking_strategy` is surfaced in the
API response's `retrieved_documents` field. The full post-chunking corpus
(base + sub-chunks, deployed and live) is **858,768** documents, verified
directly against the running bm25s index, not assumed.

---

### Decision 4: Whoosh over Elasticsearch (superseded - see note)

**Superseded**: Whoosh was later replaced by `bm25s` (`src/bm25s_service.py`)
- Whoosh's per-query disk-segment reopening made it slow even with searcher
caching, and it was returning zero results on most natural-language
queries. `bm25s` scores as a sparse matmul over a memory-mapped index
instead. `src/whoosh_service.py` was removed from the codebase.

**Why not Elasticsearch**: heavy (Java, memory), distributed setup
unnecessary at this scale. `bm25s` keeps the same benefit (pure-Python,
no distributed infra, file-based index) that made Whoosh attractive
originally, without Whoosh's correctness problems.

---

### Decision 5: Two caching layers, checked before retrieval/generation

**Literal cache** (`src/answer_cache.py`): exact `(query, retrieved_doc_ids)`
key - catches identical repeats, checked after retrieval so a corpus change
that alters which docs a query retrieves invalidates the key automatically.

**Semantic cache** (`src/semantic_cache.py`): embedding-similarity match
(`SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.92`), checked *before* retrieval -
a hit skips retrieval and generation both, not just generation. This is
also what reconciles speculative execution: a WS client can fire a
headless pipeline run on a live-caption prefix while the user is still
talking, and if the guess's embedding was close enough to the real
post-stop transcript's, the real query becomes an instant cache hit.

Hedged/declined answers are never written to either cache.

---

### Decision 6: Corpus on a Fly volume, not baked into the image

The ~4.6GB corpus (Chroma DB + bm25s index) is mounted from a Fly volume
(`fly.toml`'s `[mounts]`), not copied into the Docker image. A first
attempt baked it in directly; the resulting ~9.3GB image consistently
failed to start on Fly (confirmed via a trivial hello-world image starting
instantly on the same app/region/config, isolating the failure to image
size/pull time, not application code). Both embedding models
(`sentence-transformers` + the grounding cross-encoder) *are* baked into
the image at build time instead - the opposite tradeoff, since they're
needed at every cold start and re-downloading them each time cost 5-10+
minutes before the earlier fix.

---

## 🚀 Deployment

Live at `https://ragingoa.fly.dev` (Fly.io, `sin`/Singapore region -
`bom`/Mumbai had no capacity at deploy time). `Dockerfile` builds a
CPU-only image (torch installed from PyTorch's CPU wheel index - the
default PyPI wheel bundles unused CUDA runtime libraries) with both models
pre-downloaded at build time. `fly.toml` pins the exact deployed image
digest rather than tracking the Dockerfile directly, since `fly deploy`
silently prefers a pinned `[build] image` over the Dockerfile even when a
`--dockerfile` CLI flag says otherwise - shipping a Dockerfile change
means temporarily un-pinning, rebuilding, then re-pinning to the new
digest (documented inline in `fly.toml`).

`auto_stop_machines = false` / `min_machines_running = 1`: always-on
through the judging window, so a judge never hits a cold-start WebSocket
failure. `PYTHONUNBUFFERED=1` is set so container logs reflect real-time
progress rather than sitting in Python's stdout buffer until it flushes.

---

## 🔐 Security

- **API keys**: `.env`, never committed, never baked into a Docker image
  layer (`.dockerignore`). Loaded via `flyctl secrets set` in production.
- **`RAG_API_KEY` never reaches the browser**: `/dashboard` serves a static
  shell with no server-side templating - it used to inject the key into
  the page, which was only safe under a loopback-only bind; that was
  removed before any public deployment. Verified directly (grep the served
  page source for the key value - zero matches).
- **Public demo path instead of a shared/weakened key**: rather than
  re-exposing `RAG_API_KEY` to make the dashboard usable without setup,
  `/demo/query` and `/ws/demo` are separate, keyless, rate-limited routes
  (see Access paths above) running the identical pipeline and guardrails.
- **Rate-limit key is spoof-resistant**: keyed on Fly's `Fly-Client-IP`
  (set by Fly's edge, unspoofable by the client), not `X-Forwarded-For`
  (any caller can set that header to an arbitrary value) - caught by
  automated security review before shipping.
- **Off-topic/unsafe input gate**: see above - runs ahead of retrieval and
  generation on every access path, authenticated or demo.
- **Timing-safe key comparison**: `secrets.compare_digest`, not `==`.

---

## 📊 Monitoring

`GET /metrics` returns P50/P70/P100 latency per pipeline stage
(embedding, retrieval_dense, retrieval_sparse, generation, grounding_check,
merge_results - see `src/latency_tracker.py`), sourced from real request
traffic, not synthetic benchmarks. `GET /debug/cache_stats` (key-gated)
reports literal vs. semantic cache hit/miss counts.

For a point-in-time percentile snapshot against the live public URL,
see `benchmark/percentile_batch_results.json` and
`benchmark/run_percentile_batch.py` - rather than hardcoding specific
latency numbers into this document, where they'd inevitably go stale.

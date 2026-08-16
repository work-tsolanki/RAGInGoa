# HH Goa 2026 Task 2 — Spec Compliance & Security Audit

Audit performed against running code and live-measured behavior, not documentation claims. Every PASS below was reproduced live during this audit (commands/queries run against the actual server); every MISSING/PARTIAL was confirmed by grep/read of actual source, not inferred. The source spec document (`Markdown/spec_compliance_and_security_audit.md`) predates several changes made this session — several of its "likely MISSING" assumptions are now stale and corrected below with evidence.

## Top two findings (leading per audit instructions)

### 1. 200ms full-chain latency target: **MISSED, and by a wide margin**
Live-measured, fresh (non-cached) voice query, real-time audio pacing, full chain (STT streamed during recording + embedding + retrieval + generation + grounding):

```
Post-stop latency: 683ms
Breakdown: stt=344.0ms  embedding=13.0ms  retrieval=11.8ms  generation=280.4ms  grounding=33.0ms
```

Even a **text-only** query (no STT at all) commonly runs 200–350ms end to end on its own (generation alone is typically 165–330ms on Groq). The 200ms target is not achievable under the full-chain interpretation with an LLM-based generation step and a cloud STT call in the loop — this is a hardware/API-latency floor, not a bug to fix by Aug 22. **This needs to be a documented, known tradeoff in the submission writeup, not something a judge discovers unannounced.**

### 2. Voice pipeline: **contrary to the source doc's assumption, this is BUILT and verified working**
The audit doc assumed voice was "purely plain text" / MISSING, based on stale project history. As of this session: full voice pipeline exists and was live-tested end to end (see A.1 below) — real speech in, real transcript out, real generated answer out, in both English and Hindi/Gujarati. This is the one part of the doc's framing that needed active correction, not confirmation.

---

## Part A: Requirements traceability

### A.1 — Pipeline shape: Voice → STT → Retrieval → Generation
| Item | Status | Evidence |
|---|---|---|
| Voice input capture | **PASS** | `dashboard/index.html` — `AudioContext`+`ScriptProcessorNode` captures 16kHz mono PCM, streamed live over WebSocket (`audio_chunk` messages) |
| STT integration | **PASS** | Sarvam AI, both integrated: batch REST (`src/stt_service.py::SttService`) and realtime streaming (`src/stt_service.py::RealtimeSttSession`, `saaras:v3-realtime` model). Live pipeline (`main_app.py`, `audio_stream_start`/`audio_chunk`/`audio_stream_end`) uses the realtime path. |
| Architecture doc alignment | **PASS** | Sarvam was the documented choice and is what's implemented; no ElevenLabs code exists (correctly, per "pick one") |
| End-to-end voice test | **PASS** | Live-tested this audit with real synthesized speech, both a fresh query and a cache-hit repeat, both English and (in earlier sessions) Hindi/Gujarati — see Finding #2 above. Output is text (no TTS/spoken response) — satisfies spec's "spoken **or** text answer." |

### A.2 — Chunking strategy: spec requires more than a single naive fixed-size approach
| Item | Status | Evidence |
|---|---|---|
| Chunking strategies implemented | **MISSING** | `scripts/download_dataset.py` does not chunk documents at all — it takes MSMARCO-XI's pre-segmented passages **as-is**, one dataset passage = one retrieval unit (`data/msmarco-xi/chunks.jsonl`, 743,739 rows). `scripts/chunk_and_index.py` only embeds/indexes these pre-existing units; it contains no splitting, windowing, or size logic. |
| Multiple strategies / justified hybrid | **MISSING** | No fixed-size, semantic, or metadata-aware chunker exists anywhere in `src/` or `scripts/` |
| Overlap handling | **MISSING** | No overlap logic exists — there is nothing to overlap since documents aren't being split |
| Metadata attached and used | **PARTIAL** | Chunks carry `chunk_id`, `content`, `language`, `source`, `query_id`. Only `language` is actually used downstream (`src/retrieval.py::merge_and_rank`, ranks same-language docs first). `source`/`query_id` are stored but never read anywhere. |
| Documentation vs. implementation gap | **CONFIRMED, exactly as the audit doc warned** | `Markdown/ARCHITECTURE.md` ("Decision 3: Rich Metadata Chunks") documents a `section` (header hierarchy) field and implies real chunking — neither exists in the actual `chunks.jsonl` schema or code. This is documented-but-never-built. |

**This is the single largest spec gap found.** "Should be vast, rejects a single naive fixed-size approach" — the actual implementation doesn't even reach naive fixed-size; it's zero chunking (dataset passages used verbatim). Fixable before Aug 22 only if scoped narrowly (e.g., add one additional strategy — sentence-window chunking with overlap — as a second path for any future non-MSMARCO ingestion, since the MSMARCO corpus itself can't be meaningfully "re-chunked" further without more data prep work).

### A.3 — Latency target: <200ms full process through to final output
| Item | Status | Evidence |
|---|---|---|
| Scope resolution | Resolved per doc's own instruction: full chain (voice→answer), not retrieval-alone | — |
| Full-chain measurement | **MISSED** | See Finding #1 above: 683ms fresh voice query (STT 344ms + embedding 13ms + retrieval 12ms + generation 280ms + grounding 33ms) |
| Retrieval-alone (for contrast, not compliance) | Well under 200ms | embedding+retrieval+merge typically 15–70ms combined |
| Honest reporting | **Done here** — do not report retrieval-alone as compliance | — |

### A.4 — Latency analytics: P50/P70/P100 across a realistic sample size
| Item | Status | Evidence |
|---|---|---|
| Percentile infrastructure exists | **PASS, better than the doc assumed** | `src/latency_tracker.py::LatencyTracker.get_stats()` computes real P50/P70/P100/mean per component; live at `GET /metrics` |
| Components actually tracked from live traffic | **PARTIAL — critical gap found** | Live-checked `/metrics` after dozens of real session queries: `embedding` (51 samples), `chroma_query` (44), `bm25s_search` (44), `grounding_check` (41), `merge_results` (43) — all populated correctly. **`generation`: only 1 sample.** Root cause: the WebSocket pipeline (all real dashboard/voice traffic) calls `GenerationService.stream_generate()`, which has no `@track_latency` decorator — only the REST-only `generate()` does. The single most latency-dominant stage is effectively untracked for real traffic. |
| `run_backend_comparison.py` percentiles | **MISSING, as the doc predicted** | Reports per-query point values and simple averages only (`print_summary_table`), no percentile computation, and runs against only 5–7 fixed queries, not 30–50+ |
| Submittable artifact (table/chart) | **MISSING** | `/metrics` returns raw JSON only; no generated table or chart exists anywhere in the repo |

**Fix needed before relying on this for the submission**: add `@track_latency("generation")`-equivalent instrumentation to `stream_generate()` (or wrap its call site in `main_app.py`), then run a real 30–50 query set through the live WS pipeline to populate a genuine percentile distribution, and export it as a table/chart artifact.

### A.5 — Structured harness (not raw prompt-in/text-out)
| Item | Status | Evidence |
|---|---|---|
| Backend fallback/retry chain | **PASS** | `GenerationService._generate_with_fallback` — ordered Groq→local→Claude→extractive, real retry/error-recovery semantics (`src/generation_service.py`) |
| Modular, typed pipeline stages | **PASS** | Separate services: `EmbeddingService`, `ChromaService`, `Bm25sService`, `GenerationService`, `Guardrails`, `AnswerCache`, `SemanticCache`, `SttService`/`RealtimeSttSession` — not one large function |
| Structured I/O at service boundaries | **PASS** | `main_app.py`: `QueryRequest`/`QueryResponse`/`HealthResponse` are Pydantic `BaseModel`s; internal stage results pass as typed dicts with consistent shape (`latency_breakdown`, `retrieved_documents`, etc.) |

This item is solidly satisfied — document it plainly in the submission writeup, as the doc suggests, since a judge reading the repo structure (not just runtime output) will see this directly.

### A.6 — Guardrails: off-topic, unsafe input, hallucination, "knows when not to answer"
| Item | Status | Evidence |
|---|---|---|
| Grounding check exists | **PASS** | `Guardrails.check_grounding` — multilingual cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`), `src/guardrails.py` |
| Grounding threshold calibrated | **PASS** (doc's assumption of 0.0 is stale) | `config.py:103` — `ANSWER_CACHE_MIN_GROUNDING = 0.7`, calibrated against 38 hand-labeled examples (see `config.py` comment + `scripts/analyze_grounding_calibration.py`) |
| Grounding score actually gates the **returned answer** | **MISSING — confirmed live, not just by code reading** | `main_app.py`: the 0.7 threshold only gates whether an answer gets *cached* (`if is_valid and grounding_score >= ANSWER_CACHE_MIN_GROUNDING: answer_cache.set(...)`). It does **not** block or hedge a low-grounding answer from being returned to the user. Live-reproduced this session: a Gujarati query retrieved irrelevant Nepali/Sanskrit passages (grounding score **0.0868**), and the system still returned a confident, unhedged answer with no refusal or caveat. |
| "Knows when not to answer" | **PARTIAL** | Only `validate_answer()` gates output — rejects empty/too-short/refusal-phrase answers (`src/guardrails.py`), replacing them with "I could not find a clear answer..." This catches the model saying "I don't know," but does **not** catch the model confidently answering from ungrounded/irrelevant context, which is the more dangerous hallucination case the spec is asking about. |
| Off-topic query detection | **MISSING** | `grep -rniE "off.?topic\|moderation\|profanity\|unsafe\|abuse"` across `src/` and `main_app.py`: zero matches. No classifier, no topic-centroid check, no dedicated routing layer. |
| Unsafe/inappropriate input handling | **MISSING** | Same grep, zero matches. No input sanitization/rejection path distinct from the RAG grounding check exists. |

**Concrete, low-effort fix available before Aug 22**: wire the existing (already-computed) `grounding_score` into the returned-answer path — e.g., if `grounding_score < ANSWER_CACHE_MIN_GROUNDING`, return a hedged/refusal answer instead of the raw generation. The score is already computed on every request; this is a threshold check at one call site (`main_app.py` around the `is_valid`/cache-write block), not new infrastructure. Off-topic detection and input safety would need new code and are lower priority given remaining time.

---

## Part B: Security audit

| Check | Status | Evidence |
|---|---|---|
| B1. Full git history scan for secrets | **CLEAN** | `gitleaks`/`trufflehog` not installed in this environment; ran the doc's manual-grep fallback across full `git log -p --all` (15,264 lines) for Groq/Sarvam/OpenAI/Slack/Google key patterns and literal `api_key="..."` assignments — zero matches |
| B2. `.env` never committed | **CLEAN** | `git ls-files \| grep -iE "\.env$\|\.env\."` → only `Development/.env.example` (a template with `mock`/blank values, no real secrets) |
| B3. Current codebase hardcoded-key scan | **CLEAN** | `grep -rn "sk-\|gsk_\|AIza\|csk-"` across all `.py`/`.js`/`.ts`/`.html` (excluding `venv/`) — zero matches. Confirmed every key (`GROQ_API_KEY`, `SARVAM_API_KEY`, `CEREBRAS_API_KEY`, `ANTHROPIC_API_KEY`, `RAG_API_KEY`) is sourced via `os.getenv()` in `config.py`, never a literal. |
| B4. `.gitignore` present from first commit | **CLEAN** | `.gitignore` (with `.env` rule) is part of the very first commit (`bd18cd1`, "Get RAG quick-start working end-to-end on localhost") — no early window existed where `.env` could have been committed unignored |
| B5. Client-side key exposure | **CLEAN, with one documented-and-justified exception** | Groq/Sarvam/Cerebras/Anthropic keys: zero references in `dashboard/index.html`, all server-side only. `RAG_API_KEY` (the app's own auth token, not a third-party secret) **is** injected into the dashboard HTML at request time — this is intentional and safe *only* because the server binds to `127.0.0.1` exclusively (verified: `main_app.py:999`, `uvicorn.run(..., host="127.0.0.1", ...)`), so the dashboard is never reachable by anyone who couldn't already reach the API directly. No Jupyter notebooks or CI/CD config files exist in the repo (nothing to check there). |
| B5. README/docs leftover keys | **CLEAN** | No real-looking key strings found in any `.md` file |

**Part B overall: clean.** No rotation or history rewrite needed.

---

## Part C: Submission logistics

| Item | Status | Evidence |
|---|---|---|
| GitHub repo public | **PASS** | Unauthenticated `GET https://api.github.com/repos/work-tsolanki/RAGInGoa` → `200` (private repos 404 to unauthenticated requests) |
| Live working link (deployed, non-localhost) | **MISSING** | Server is bound to `127.0.0.1` only (see B5) — by design, for the `RAG_API_KEY`-in-dashboard security model. **There is currently no publicly reachable deployment.** This needs either a deploy (with the loopback-only dashboard-key assumption revisited for a public host) or explicit scoping in the submission that the live link is a local/tunneled demo, before Aug 22. |
| Team/process video (90s) | **Cannot verify — outside this repo/codebase** | Needs manual confirmation from the team |
| Demo video, and does it imply capabilities not in the live link | **Cannot verify content**, but flag: given the STT pipeline **is** real (Finding #2) and the live link **doesn't exist** yet (item above), a demo video showing voice functionality would currently be showing something not reachable at any public URL — resolve the live-link gap before this becomes a credibility risk |
| Hashtag: `#RAGInGoa` (not `#FrameInGoa`) | **Cannot verify** — no posts exist in this repo to check; flagging as a reminder per the spec doc |
| Posts on Instagram/X/LinkedIn, per-member, one public Instagram | **Cannot verify** — same as above, human/social-media-side confirmation needed |

---

## Summary of what needs action before Aug 22, ranked

1. **Live public deployment** — currently loopback-only; no public URL exists. Needed for both the "live working link" requirement and to make the demo video's claims verifiable.
2. **Grounding score should gate the answer, not just the cache** — cheap, already-computed data, one threshold check away from actually satisfying "knows when not to answer" for the hallucination case (currently only catches explicit refusal phrases).
3. **Chunking strategy** — the largest structural gap. At minimum, document plainly in the submission that MSMARCO-XI's passages are used as pre-segmented retrieval units rather than claiming a chunking strategy that isn't there; better, add one real second strategy if time allows.
4. **200ms target** — cannot be met under the full-chain interpretation with current architecture (cloud STT + LLM generation). Document as a known, explained tradeoff rather than leaving it to be discovered.
5. **`generation` stage missing from `/metrics` percentiles** — one-line instrumentation fix (`stream_generate()` needs the same tracking `generate()` has), then run a real 30–50 query batch to produce a submittable P50/P70/P100 artifact.
6. Off-topic / unsafe-input detection — genuinely missing, lowest priority given remaining time and lower spec weight than the above.

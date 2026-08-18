# Mercury 2 promotion decision

**Result: benchmarked, NOT promoted.** `GENERATION_BACKEND_ORDER` stays `["groq", "local", "claude"]`. Mercury is fully integrated (`_generate_mercury_stream`, unit-tested) and reachable via `GENERATION_BACKEND_ORDER_OVERRIDE=mercury,groq,local,claude` for future re-evaluation, but not in the live default path.

## Benchmark (7 queries: 5 English + 1 Hindi + 1 Gujarati, `reasoning_effort="instant"`)

| Backend | Avg total_ms | Avg generation_ms | Avg grounding | Valid % |
|---|---|---|---|---|
| groq    | 591.6 | 234.5 | 0.675 | 100% |
| mercury | 676.5 | 650.7 | 0.509 | 100% |
| local   | 540.7 | 518.5 | 0.651 | 100% |

Full results: `benchmark/results.json`.

## Gate evaluation (master_implementation_prompt-mercury2.md Priority 3.6)

1. **Average total_ms meaningfully lower than Groq: FAIL.** Mercury is slower (676.5ms vs 591.6ms), not faster — contradicts Mercury's own >1000 tok/s claim. `reasoning_effort="instant"` (the fastest tier, per Inception's docs) was used throughout. Most likely explanation: at our answer lengths (15-183 tokens), diffusion's fixed per-request overhead (network hop to a different provider, denoising-step floor) doesn't amortize the way it would on longer generations - the same shape of finding as the speculative-decoding investigation earlier in this project (short-generation workloads don't have enough runway for architectural speedups with fixed overhead to pay off).
2. **Average grounding_score comparable or better: FAIL on raw numbers (0.509 vs 0.675), with an important caveat.** Manual review (below) shows part of this gap is Mercury correctly giving a low-scoring honest refusal where Groq gave a high-scoring hallucination (Gujarati voter-ID query) - so the raw average understates Mercury's actual answer quality here. Still fails the gate as literally stated.
3. **Manual review, Hindi/Gujarati - no regressions found.** Hindi: all three backends stayed correctly in-language with coherent, comparably detailed answers. Gujarati: Mercury's answer was the *most* trustworthy of the three (honest refusal vs. two hallucinations) despite scoring lowest - a genuinely interesting finding, opposite of the a priori "diffusion model = unfamiliar fluent-but-wrong failure mode" concern the master prompt flagged.
4. **is_valid rate matches Groq: PASS.** 100% for all three backends.

## Decision

Gate 1 (speed) fails outright and is the deciding factor for a project whose explicit goal is lowest latency - Mercury does not deliver its core promised advantage on this system's short-answer RAG workload. Gate 2 fails on the literal numbers even accounting for the calibration nuance. Per the master prompt's rule ("if any of these fail, keep Groq primary, keep Mercury integrated-but-not-live, document why") - Groq remains primary.

**Worth revisiting if:** Inception ships a lower-overhead endpoint tier, answer lengths grow substantially (where diffusion's fixed cost would amortize better), or Mercury's honest-refusal behavior turns out to matter more than raw speed for a future use case.

## Addendum (Priority 4): output token cap tuning

Tested whether `max_tokens` maps onto latency for Mercury the way it does for autoregressive backends (same prompt, 3 repeats per cap, `"What is the minimum wage"` - a query known to produce a longer answer):

| Backend | cap=50 | cap=100 | cap=200 |
|---|---|---|---|
| groq (autoregressive) | 190ms / 25 words | 242ms / 59 words | 271ms / 71 words |
| mercury (diffusion) | 754ms / 27 words | 584ms / 55 words | 967ms / 61 words |

Groq scales cleanly and monotonically with `max_tokens` - the expected autoregressive shape, confirming "shorten the output" is a real lever there. Mercury does not: `cap=100` was faster than `cap=50` in this run, `cap=200` was slowest, and per-call variance was much higher (541-1513ms vs Groq's 128-292ms). Confirms the plan's hypothesis - Mercury's latency isn't dominated by output length, so `max_tokens` tuning isn't a reliable lever for it specifically. This is an additional data point against Mercury for latency-sensitive serving, independent of the Priority 3 gate result above.

## Addendum 2: raw network RTT isolation (rules out network geography as the cause)

Isolated per the "is this just network geography" hypothesis - raw `requests.Session().get(base_url)` RTT, cold (fresh TLS handshake) vs warm (connection reused, same as what the SDK clients do across repeated calls in-process):

| Endpoint | Cold | Warm (steady-state) |
|---|---|---|
| groq | 442ms | ~260-290ms (one 928ms blip in 10 samples, otherwise stable) |
| api.inceptionlabs.ai (Mercury) | 461ms | **~13-25ms** |
| api.cerebras.ai | 486ms | ~317-319ms (very stable) |

This cuts the opposite way from what "network geography explains it" would predict: Inception's warm RTT is the *fastest* of the three, not the slowest, yet Mercury's actual generation calls were the slowest (Addendum 1, and the Priority 3 benchmark). That rules out network transport as the explanation and points squarely at inference-time compute cost on Mercury's side (the diffusion denoising process itself) as the real bottleneck - not something a closer region or a backend swap fixes. This strengthens, rather than undermines, the "not promoted" decision above.

Cerebras's own ~318ms warm RTT is real and independently high (worse than Groq's), a separate contributing factor for whenever its billing block (see `cerebras_status.md`) is resolved and a live benchmark can actually run.

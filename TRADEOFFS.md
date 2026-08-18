# Tradeoffs

Honest record of the constraints we hit and the calls we made, for judges and for us.

## 1. Limited corpus size → more question declines

**Constraint:** Fly's persistent volume has a real, paid size limit. Growing it costs money and we couldn't justify an unbounded expansion for a hackathon build.

**What we did:** The dataset (`ai4bharat/MSMARCO-XI`) has ~98,000 rows per language available, but we only ingested a slice (5,000-7,000 rows/language, later expanded once) to stay inside a volume we could actually afford: 20GB, not 100GB+.

**Consequence:** The corpus doesn't have real content for every general-knowledge topic a user might ask. The off-topic gate correctly declines those instead of guessing, so some reasonable questions get declined, not because the *system* is broken, but because the *dataset we could afford to host* doesn't cover them. Confirmed directly: bypassed the gate and checked retrieval for declined topics. For most of them, the corpus genuinely has no good passage, not just a gate that's too strict.

## 2. No GPU on Fly → slower grounding check

**Constraint:** Fly's standard machines are CPU-only; GPU instances are a different, pricier tier we didn't provision.

**What we did:** The grounding check (a cross-encoder scoring answer-vs-context) runs on CPU in production. Measured live across 137 real requests (see the latency table below): **grounding_check P50 ≈ 324ms, P70 ≈ 400ms**. This one stage alone is already past the spec's 200ms target before anything else in the pipeline runs.

**Consequence:** This is the single biggest latency cost in the pipeline, and it's a direct GPU-vs-CPU tradeoff, not a code inefficiency: the same cross-encoder call is near-instant on a GPU. We reduced its cost once already (cutting scored docs from 5 to fewer), but a CPU can't fully close this gap.

## 3. No Fly region in India → extra network RTT

**Constraint:** Fly's Mumbai region (`bom`) had zero VM capacity at deploy time, confirmed by trying to deploy there directly.

**What we did:** Deployed to Singapore (`sin`) instead, the closest region with capacity.

**Consequence:** Every request (a judge or user connecting from India, and this app's own calls out to the Groq inference API) crosses an extra international hop instead of staying in-region. That RTT is baked into every number below; it's not something application code can optimize away.

## 4. Off-topic gate kept narrow, on purpose

**Constraint:** We tested loosening the gate to answer more general-knowledge questions. It worked for some topics, but for others (e.g. "capital of Portugal"), the corpus had *keyword-adjacent but wrong* content, and the system answered **confidently and incorrectly** instead of declining.

**What we did:** Reverted the broad loosening. Only added gate exceptions for topics we individually verified the corpus can genuinely answer correctly (e.g. "heart pumping blood"). Left every other general-knowledge topic declined.

**Consequence:** More declines than a fully-open gate would give. That's the deliberate trade: **a wrong confident answer is worse than an honest "I don't know."** The spec explicitly asks the system to "know when not to answer"; this is that requirement in practice, not a gap.

---

## Latency analytics (P50 / P70 / P100)

Per the spec's requirement to measure across a reasonable number of queries, not a single best-case run: **160 queries** against the live production deployment, right after a clean restart (so no cached answers inflate the numbers). 136 of those genuinely ran the full pipeline; 24 hit the answer cache mid-run and are excluded below so the numbers reflect real, uncached latency.

**End-to-end (client-measured, includes network round-trip):**

| Metric | P50 | P70 | P100 |
|---|---|---|---|
| Wall clock | 528.5ms | 599.8ms | 2081.8ms |
| Server-reported total | 459.0ms | 529.9ms | 2012.4ms |

**Per-stage (server-side, n=137):**

| Stage | P50 | P70 | P100 |
|---|---|---|---|
| Embedding | 27.6ms | 29.3ms | 2416.5ms* |
| Dense retrieval (Chroma) | 16.3ms | 21.9ms | 62129.9ms* |
| Sparse retrieval (BM25) | 10.0ms | 14.9ms | 43.2ms |
| **Grounding check (cross-encoder)** | **323.7ms** | **399.5ms** | 1876.5ms |
| Generation (Groq) | 78.4ms | 81.4ms | 396.0ms |

*P100 outliers on embedding/Chroma are a single cold-start data point (first query after the restart, before OS page cache warmed up on the 5GB ChromaDB file), not representative of steady-state.

**Read honestly:** the full chain is over the spec's 200ms target at P50 (~459-528ms), and it isn't close. **The grounding check alone (P50 323.7ms) accounts for the majority of it**, that's tradeoff #2 (no GPU) in direct numbers. Retrieval itself is fast (P50 ~26ms combined dense+sparse) and was never the bottleneck; the CPU-bound cross-encoder is. Region latency (tradeoff #3) adds on top via the wall-vs-total gap (~70-100ms of network RTT). We're stating this plainly rather than cherry-picking a best-case run: a GPU instance would likely bring this under 200ms; that's a cost decision beyond this project's budget, not an unknown engineering fix.

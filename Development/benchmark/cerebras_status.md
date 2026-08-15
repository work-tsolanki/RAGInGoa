# Cerebras integration status: blocked, not benchmarked live

**Code is fully integrated and unit-tested** (`_generate_cerebras_stream`, registered in the fallback chain, reachable via `GENERATION_BACKEND_ORDER_OVERRIDE`). Live benchmarking is blocked on account setup, not code.

## What happened

1. The plan called for testing Cerebras with Llama-3.1-8B (same weights as Groq, silicon-only speed comparison). That model isn't available on this account - `client.models.list()` returned only `gemma-4-31b`, `zai-glm-4.7`, `gpt-oss-120b`.
2. Tried `gpt-oss-120b` instead (user's call, since the narrow same-weights test wasn't possible) - this changes the test from "speed only" to "speed + accuracy, different/larger model."
3. All three available models return `402 Payment Required` on this account, despite the free-tier documentation (1M tokens/day, no credit card) found via search. Either that's changed, or this account needs additional verification.

## To unblock

Resolve billing/verification at cloud.cerebras.ai, then:
```
python benchmark/run_backend_comparison.py --backends groq,mercury,cerebras,local
```

## Current config

`CEREBRAS_MODEL = "gpt-oss-120b"` in `config.py` - update if a different model becomes available/preferred once billing is resolved.

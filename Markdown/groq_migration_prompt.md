# Task: Add Groq API as primary generation backend, with local llama.cpp and Claude API as fallbacks

## Context

This is a voice-enabled RAG system (Task 2, HH Goa 2026 submission). Current architecture:

- **Retrieval**: hybrid dense (ChromaDB) + sparse (bm25s) retrieval, RRF fusion, ~40ms total — already optimized, do not touch.
- **Answer cache**: `src/answer_cache.py` — LRU cache keyed on normalized query + sorted doc IDs, TTL 24h. Already wired into `/query` and the WebSocket handler in `main_app.py`. Do not touch the cache logic itself, only where it sits in the new flow.
- **Generation**: currently local llama.cpp via llama-cpp-python, `Llama-3.1-8B-Instruct` Q4_K_M, `create_chat_completion`, `max_tokens=200`, `temperature=0.3`. Falls back to Claude API, then raw passage return, if local model is unavailable.
- **Guardrails**: `guardrails.py::check_grounding` (cross-encoder mmarco-mMiniLMv2-L12-H384-v1, scores answer against retrieved docs) and `guardrails.py::validate_answer` (rejects empty/too-short/refusal answers). These run on the final answer text regardless of which backend produced it — do not modify their internals.
- **Streaming**: tokens are streamed to the client (WebSocket) as they're generated; grounding check and cache write happen after the full answer is assembled server-side, using a buffered copy of the streamed text.

## Goal

Local generation currently runs at ~130 tokens/sec (700ms-1.5s for typical 33-183 token answers). Groq's `llama-3.1-8b-instant` runs the same model at ~840 tokens/sec, hosted, at negligible per-token cost. Swap it in as the **primary** generation path, keep local llama.cpp as the fallback (instead of the primary), and keep Claude API as the final fallback. No changes to retrieval, caching logic placement rules, or guardrails logic.

## Steps

### 1. Add dependency and config

- Add `groq` to `requirements.txt` (official Python SDK, OpenAI-compatible).
- Add to `config.py`:
  - `GROQ_API_KEY` (read from environment variable `GROQ_API_KEY`, never hardcoded)
  - `GROQ_MODEL = "llama-3.1-8b-instant"`
  - `GROQ_MAX_TOKENS = 200` (match existing local cap)
  - `GROQ_TEMPERATURE = 0.3` (match existing local setting)
  - `GENERATION_BACKEND_ORDER = ["groq", "local", "claude"]` — explicit ordered list so backend priority is configurable without code changes.
- Add `GROQ_API_KEY` to `.env.example` with a placeholder and a comment linking to https://console.groq.com/keys.

### 2. Implement the Groq generation function

In `generation_service.py`, add a new function alongside the existing local generation function, matching its exact signature and streaming behavior (yields text deltas, same as the current local streaming implementation):

```python
from groq import Groq, GroqError

groq_client = Groq(api_key=config.GROQ_API_KEY) if config.GROQ_API_KEY else None

def generate_groq(prompt: str, max_tokens: int = None, temperature: float = None):
    """
    Stream generation from Groq's llama-3.1-8b-instant.
    Raises GroqError on failure so the caller can fall through to the next backend.
    """
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not configured")

    stream = groq_client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens or config.GROQ_MAX_TOKENS,
        temperature=temperature or config.GROQ_TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

Use the exact same `_build_prompt` output as the local path — do not create a separate prompt template for Groq. The instruction template, retrieved-passage formatting, and the language-matching instruction (the one already added to fix the English-query/Gujarati-answer bug) must be identical across all three backends, so answer quality and grounding behavior stay comparable regardless of which backend served the request.

### 3. Build the fallback chain

Refactor the existing single-backend call site into an ordered-fallback wrapper. Use `config.GENERATION_BACKEND_ORDER` to drive the sequence rather than hardcoding it, so the order can be changed via config alone later:

```python
def generate_with_fallback(prompt: str, max_tokens: int = None, temperature: float = None):
    backends = {
        "groq": generate_groq,
        "local": generate_local,   # existing local llama.cpp function
        "claude": generate_claude, # existing Claude API fallback function
    }
    last_error = None
    for name in config.GENERATION_BACKEND_ORDER:
        try:
            full_text = ""
            for delta in backends[name](prompt, max_tokens, temperature):
                full_text += delta
                yield {"delta": delta, "backend": name}
            return  # success, stop trying further backends
        except Exception as e:
            last_error = e
            log.warning(f"Generation backend '{name}' failed: {e}. Trying next.")
            continue
    raise RuntimeError(f"All generation backends failed. Last error: {last_error}")
```

Notes:
- If a backend fails **mid-stream** (partial tokens already sent to the client before an error), do not silently retry with a different backend for that same request — log it, let the partial response go to `validate_answer` as-is, and let normal empty/short-answer rejection handle it. Silently discarding a partial stream and starting over on a different backend risks sending the client two different partial answers back to back.
- Only retry on connection/timeout/rate-limit errors before any tokens have been yielded. Once streaming has started, commit to that backend for the request.

### 4. Wire into `main_app.py`

- Replace the current direct call to the local generation function (in both the `/query` endpoint and the WebSocket handler) with `generate_with_fallback`.
- Keep the existing flow order exactly as-is: answer cache check happens **before** calling any generation backend (after retrieval, using doc IDs for the cache key) — this doesn't change. Generation backend selection only matters on a cache **miss**.
- Buffer the full streamed text server-side (as already implemented) for `check_grounding` and `validate_answer` after the stream completes — this logic doesn't change, it just now receives text from whichever backend actually served the request.
- Add `"backend"` to the response payload alongside the existing `"cache_hit"` field, so you can see in logs/responses which backend served each request (useful for debugging and for measuring how often it falls through past Groq).

### 5. Testing checklist

- [ ] Unit test `generate_groq` with a mocked Groq client: confirm it yields deltas in the same shape as `generate_local`.
- [ ] Unit test `generate_with_fallback`: force `generate_groq` to raise before yielding anything, confirm it falls through to `generate_local` cleanly.
- [ ] Integration test: run your existing 3 benchmark queries ("How to apply for a passport", "What is a corporation?", "income tax filing deadline") against the live Groq backend. Record and compare against your existing local-backend numbers:
  - prompt tokens / answer tokens / total generation time
  - grounding score (should be comparable to local runs, since it's the same model weights)
  - manually check answer text for the same language-match and coherence you validated in the k=5 vs k=10 comparison
- [ ] Confirm answer cache still works correctly on top of Groq-served answers: same query twice should hit cache on the second call regardless of which backend served the first.
- [ ] Kill-switch test: temporarily set an invalid `GROQ_API_KEY`, confirm the request falls through to local llama.cpp without the request failing or hanging.
- [ ] Load-shape test: confirm streaming still delivers the same perceived first-token latency improvement as before, now sourced from Groq instead of local.

### 6. What NOT to change

- Retrieval pipeline (`chroma_service.py`, `bm25s` sparse retrieval, `retrieval.py::merge_and_rank` / RRF fusion) — already optimized, out of scope.
- `answer_cache.py` internals — key construction, LRU/TTL logic, min-grounding-to-cache threshold — unchanged.
- `guardrails.py` — grounding and validation logic unchanged; it should treat all three backends' output identically.
- The prompt template / instruction wrapper in `_build_prompt`, including the language-matching instruction — must stay identical across backends.

## Environment setup (manual step, not code)

1. Sign up at https://console.groq.com, generate an API key.
2. Add `GROQ_API_KEY=<key>` to the local `.env` file (not committed).
3. Confirm the free tier is sufficient for initial testing (30 req/min, 1,000-14,400 req/day) before any paid usage.

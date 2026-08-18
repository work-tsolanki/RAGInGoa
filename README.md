# RAGInGoa

Voice-enabled RAG system built for HH Goa 2026 Task 2. Speak or type a question → transcription (Sarvam STT) → hybrid dense (ChromaDB) + sparse (BM25) retrieval → grounded answer generation (Groq, Claude fallback) → guardrail checks (off-topic, unsafe, hallucination/grounding) before the answer is returned.

**Live:** https://ragingoa.fly.dev
**Dashboard:** https://ragingoa.fly.dev/dashboard

> **Latency is above the spec's 200ms target.** Read [`TRADEOFFS.md`](TRADEOFFS.md) first — it explains why (CPU-only hosting, no India region, corpus size) with real measured numbers, not guesses.

## Pipeline

```
Voice/text input → Speech-to-text → Chunking/Retrieval (vector DB) → Guardrails → Answer generation
```

- **STT:** Sarvam (`src/stt_service.py`) — batch + realtime streaming
- **Chunking:** fixed-overlap + semantic-boundary sub-chunking on top of whole-passage entries (`src/chunking/`), metadata-aware dedup at retrieval time
- **Retrieval:** hybrid dense (ChromaDB) + sparse (BM25s), weighted fusion (`src/retrieval.py`)
- **Generation:** Groq primary, Claude fallback, structured backend chain with timeouts/retries (`src/generation_service.py`)
- **Guardrails:** unsafe-input gate, off-topic gate, grounding/hallucination check (`src/guardrails.py`)

See [`TRADEOFFS.md`](TRADEOFFS.md) for the real constraints hit (corpus size, no GPU, no India region, gate tuning) and measured P50/P70/P100 latency.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY, ANTHROPIC_API_KEY, SARVAM_API_KEY, RAG_API_KEY
uvicorn main_app:app --reload
```

Dataset: [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — see `scripts/download_dataset.py` and `scripts/expand_dataset.py` to build the corpus, then `scripts/chunk_and_index.py` + `scripts/build_bm25s_index.py` to index it.

## Tests

```bash
pytest tests/
```

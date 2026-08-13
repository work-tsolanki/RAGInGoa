# HH Goa 2026: Voice-Enabled RAG System

**Task**: Build a voice-enabled Retrieval-Augmented Generation (RAG) system that transcribes speech, retrieves relevant context from MSMARCO-XI dataset, and generates grounded answers in <200ms.

**Stack**: Python + FastAPI + Sarvam (STT) + AI4Bharat (Embeddings) + Pinecone + Whoosh + Llama 7B (local) + Claude API (fallback)

---

## 📋 Project Structure

```
hhgoa-rag/
├── README.md                          # This file
├── SETUP.md                           # Installation & environment setup
├── ARCHITECTURE.md                    # System design & data flow
├── IMPLEMENTATION_GUIDE.md            # Step-by-step build instructions
├── TESTING.md                         # Testing & validation procedures
├── DEPLOYMENT.md                      # VPS deployment guide
│
├── config.py                          # Configuration & API keys
├── requirements.txt                   # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── embedding_service.py          # AI4Bharat embedding model
│   ├── pinecone_service.py           # Pinecone vector DB wrapper
│   ├── whoosh_service.py             # BM25 search wrapper
│   ├── retrieval.py                  # Merge & ranking logic
│   ├── stt_service.py                # Sarvam STT integration
│   ├── generation_service.py         # Llama + Claude LLM wrapper
│   ├── guardrails.py                 # Grounding checks & validation
│   ├── latency_tracker.py            # Latency measurement & analytics
│   └── main.py                        # FastAPI app
│
├── data/
│   ├── test_chunks.json              # Dummy test data
│   └── sample_queries.json           # Test queries for evaluation
│
├── tests/
│   ├── test_embedding.py             # Test embedding service
│   ├── test_retrieval.py             # Test retrieval pipeline
│   ├── test_integration.py           # End-to-end tests
│   └── load_test.py                  # Latency measurement
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # MSMARCO-XI analysis
│   ├── 02_chunking_strategy.ipynb    # Chunking experiments
│   └── 03_latency_profiling.ipynb    # Latency optimization
│
└── scripts/
    ├── download_dataset.py           # Download MSMARCO-XI
    ├── chunk_and_index.py            # Offline indexing
    ├── measure_latency.py            # Run latency benchmarks
    └── deploy.sh                     # Deployment script
```

---

## 🎯 Quick Start (5 minutes)

### 1. Clone & Setup
```bash
git clone <your-repo>
cd hhgoa-rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure
```bash
cp config.example.py config.py
# Edit config.py with your API keys:
# - PINECONE_API_KEY
# - SARVAM_API_KEY
# - ANTHROPIC_API_KEY
```

### 3. Test Retrieval Pipeline (Mock Data)
```bash
python -m pytest tests/test_retrieval.py -v
# Expected: All tests pass in <100ms
```

### 4. Start Local Server
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
# Visit http://localhost:8000/docs for interactive API docs
```

### 5. Test End-to-End
```bash
python tests/test_integration.py
# Expected: Retrieve results + generate answer in <200ms
```

---

## 📊 Implementation Timeline

| Day | Phase | Deliverable | Status |
|-----|-------|-------------|--------|
| 1-2 | **Retrieval Pipeline** | Dense + BM25 retrieval working | ⏳ |
| 3-4 | **Chunking & Indexing** | MSMARCO-XI indexed in Pinecone + Whoosh | ⏳ |
| 5-6 | **Generation & LLM** | Local Llama + Claude API fallback | ⏳ |
| 7 | **Integration & Guardrails** | Full pipeline orchestrated | ⏳ |
| 8 | **Testing & Optimization** | P50/P70/P100 latency measured | ⏳ |
| 9 | **Demo & Submission** | GitHub repo + live link + videos | ⏳ |

---

## 🏗️ Architecture Overview

```
User speaks
    ↓
[Sarvam STT] ~50ms
    ↓
[Query Preprocessing] ~5ms
    ↓
    ├─→ [Dense Embedding (AI4Bharat)] ~15ms → [Pinecone Query] ~20ms
    ├─→ [BM25 Search (Whoosh)] ~20ms
    ↓
[Merge & Rank Results] ~5ms
    ↓
[Select LLM] (fast: Llama, quality: Claude)
    ↓
[Generate Answer] ~100-150ms
    ↓
[Grounding Check] ~20ms
    ↓
Return answer (if grounded) or refine (if needed)
    ↓
Total latency: 170-220ms ✓
```

**Latency Budget**:
- STT: 30-50ms
- Retrieval: 40-50ms
- Generation: 100-120ms
- Guardrails: 20ms
- **Total: <200ms target**

---

## 🔑 Key Design Decisions

### 1. Hybrid Retrieval (Dense + BM25)
- **Why**: Catches both semantic matches ("What does identity mean?") and keyword matches ("Aadhaar registration form")
- **Trade-off**: Slightly slower (~40-50ms) than pure dense, but better quality
- **Alternative**: ColBERT (better quality, but too slow for 200ms budget)

### 2. Adaptive LLM Strategy
- **Fast path**: Local Llama 7B (~100ms) → default for all queries
- **Quality path**: Claude API (~200ms) → only if grounding check fails
- **Why**: Hits latency budget 80% of the time, ensures quality when needed

### 3. Modular Services
- Each service (embedding, retrieval, generation) is independent
- Easy to swap implementations (Pinecone → Milvus, Llama → Mistral)
- Testable in isolation

### 4. Rich Metadata
- Store language, section hierarchy, confidence score per chunk
- Enables filtering: "Only search in Hindi docs"
- Helps re-ranking: "Prioritize high-confidence chunks"

---

## 📈 Latency Targets

**P50 (median)**: 150-170ms
**P70 (70th percentile)**: 180-200ms
**P100 (worst case)**: 220-300ms (when API refinement needed)

---

## 🧪 Testing Strategy

### Unit Tests
```bash
pytest tests/test_embedding.py        # Embedding service
pytest tests/test_retrieval.py        # Retrieval pipeline
pytest tests/test_generation.py       # LLM generation
```

### Integration Tests
```bash
pytest tests/test_integration.py      # Full pipeline
python tests/load_test.py            # 50 queries, measure P50/P70/P100
```

### Manual Testing
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "What is Aadhaar?"}'
```

---

## 🚀 Next Steps

1. **Read SETUP.md** → Install dependencies & configure environment
2. **Read ARCHITECTURE.md** → Understand system design
3. **Follow IMPLEMENTATION_GUIDE.md** → Build step-by-step
4. **Run TESTING.md** → Validate each component
5. **Deploy via DEPLOYMENT.md** → Get live endpoint

---

## 📞 Troubleshooting

### "Embedding model not found"
→ Check internet connection, wait for download to complete
→ Verify GPU memory (needs ~4GB for AI4Bharat)

### "Pinecone connection failed"
→ Verify API key in config.py
→ Check internet connection from VPS

### "Whoosh index corrupted"
→ Delete `whoosh_index/` directory, re-run indexing

### "Latency exceeds 200ms"
→ Run `python notebooks/03_latency_profiling.ipynb` to identify bottleneck
→ Common fixes: batch embedding, cache Whoosh results, reduce top-k

---

## 📄 License & Attribution

**Dataset**: MSMARCO-XI (AI4Bharat)
**Models**: AI4Bharat IndicBERT, Meta Llama 2
**Services**: Sarvam (STT), Anthropic (Claude API), Pinecone (vector DB)

---

## 🎯 Success Criteria

- [ ] Retrieval pipeline latency <50ms (P50)
- [ ] Generation latency <150ms (P50)
- [ ] Total end-to-end latency <200ms (P50)
- [ ] Grounding check success rate >85%
- [ ] GitHub repo with clean code & docs
- [ ] Live endpoint responding in <300ms
- [ ] Two videos (process + demo) uploaded to Instagram/X/LinkedIn
- [ ] Submitted to leaderboard

---

**Status**: 🟡 In Development (Day 1 of 9)
**Last Updated**: August 13, 2026

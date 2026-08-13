# Testing & Validation Guide

Complete testing strategy to validate each component and measure latency.

---

## Unit Tests

### Test Embedding Service
```bash
pytest tests/test_embedding.py -v

# Expected:
# test_load_model PASSED
# test_single_embedding PASSED  
# test_batch_embedding PASSED
# test_dimension PASSED
```

### Test Retrieval
```bash
pytest tests/test_retrieval.py -v

# Expected:
# test_whoosh_retrieval PASSED
# test_dense_retrieval PASSED
# test_merge_results PASSED
# test_full_pipeline PASSED [85ms latency]
```

### Test LLM Services
```bash
pytest tests/test_generation.py -v

# Expected:
# test_llama_generation PASSED [120ms]
# test_claude_generation PASSED [200ms]
# test_grounding_check PASSED
```

---

## Integration Tests

### Full End-to-End Test
```bash
pytest tests/test_integration.py -v

# Test all components together
# Expected: Full pipeline <250ms
```

### With Real Audio
```bash
python tests/test_with_audio.py \
  --audio_file test_audio.wav \
  --query_text "What is Aadhaar?"

# Expected:
# STT: 45ms
# Retrieval: 50ms
# Generation: 120ms
# Grounding: 20ms
# Total: 235ms ✓
```

---

## Latency Benchmarking

### Measure P50/P70/P100
```bash
python tests/load_test.py \
  --num_queries 50 \
  --output latency_report.json

# Runs 50 queries, measures all components
# Outputs:
# {
#   "stt": {"p50": 42, "p70": 48, "p100": 62},
#   "retrieval": {"p50": 48, "p70": 52, "p100": 78},
#   "generation": {"p50": 115, "p70": 140, "p100": 180},
#   "total": {"p50": 175, "p70": 195, "p100": 240}
# }
```

### Generate Report
```bash
python tests/generate_report.py --input latency_report.json

# Generates HTML report with charts
# Open: latency_report.html
```

---

## Manual Testing

### Start Development Server
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Interactive docs: http://localhost:8000/docs
```

### Test via API
```bash
# Text query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "What is Aadhaar?"}'

# Audio query
curl -X POST http://localhost:8000/query_audio \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@test_audio.wav" \
  -F "language=en"
```

### Test Health Endpoint
```bash
curl http://localhost:8000/health

# Expected:
# {"status": "ok", "models_loaded": true}
```

---

## Regression Testing

### Create Test Suite
```bash
# tests/regression_queries.json
[
  {"query": "What is Aadhaar?", "expected_docs": ["doc_001"]},
  {"query": "आधार क्या है?", "expected_docs": ["doc_002"]},
  {"query": "How to apply for Aadhaar", "expected_docs": ["doc_003"]},
]

# Run regression
python tests/regression_test.py --suite regression_queries.json
```

---

## Quality Metrics

### Grounding Check Accuracy
```python
# tests/test_grounding.py
def test_grounding_accuracy():
    queries = [
        ("What is Aadhaar?", True),   # Should be grounded
        ("Tell me a joke", False),    # Should NOT be grounded
    ]
    
    for query, should_ground in queries:
        results = retrieve(query)
        answer = generate(query, results)
        is_grounded = check_grounding(answer, results)
        assert is_grounded == should_ground
```

### Retrieval Quality
```python
def test_retrieval_quality():
    # Check top result relevance
    query = "Aadhaar benefits"
    results = retrieve(query)
    
    assert results[0]["doc_id"] in ["doc_004", "doc_005"]
    assert results[0]["final_score"] > 0.7
```

---

## Performance Profiling

### Profile Embedding Service
```bash
python -m cProfile -s cumtime src/embedding_service.py

# Shows CPU time per function
# Look for bottlenecks
```

### Memory Profiling
```bash
python -m memory_profiler src/main.py

# Check for memory leaks
# Typical: 2-3GB for models loaded
```

### GPU Utilization
```bash
watch -n 1 nvidia-smi

# Monitor GPU usage while running queries
# Llama should use ~8GB VRAM
```

---

## Failure Cases

### Test STT Failure
```python
def test_stt_failure():
    with patch('sarvam_client.transcribe', side_effect=Exception("API error")):
        response = query_audio(audio_bytes)
        assert response["status"] == "error"
        assert "fallback_text" in response  # Should have fallback
```

### Test Retrieval Failure
```python
def test_retrieval_failure():
    with patch('pinecone.query', side_effect=ConnectionError()):
        response = query("test")
        assert response["status"] == "partial"
        # Should still return BM25 results
```

### Test Generation Failure
```python
def test_generation_failure():
    with patch('llama_inference', side_effect=OOMError()):
        response = query("complex question")
        # Should fallback to shorter answer
        assert response["status"] == "ok"
```

---

## Checklist Before Submission

- [ ] All unit tests pass: `pytest tests/ -v`
- [ ] Latency P50 < 180ms
- [ ] Latency P70 < 200ms
- [ ] Grounding accuracy > 85%
- [ ] Handles multilingual input (en, hi, ta)
- [ ] Graceful failure (no 500 errors)
- [ ] No sensitive data in logs
- [ ] API docs work: `/docs`
- [ ] Health check passes
- [ ] Load test completes without crashes (50 queries)

---

## Running Full Test Suite

```bash
# Setup
source venv/bin/activate
pip install -r requirements.txt

# Unit tests
pytest tests/test_*.py -v

# Integration test
pytest tests/test_integration.py -v --tb=short

# Latency benchmark
python tests/load_test.py --num_queries 50 --output results.json

# Generate report
python tests/generate_report.py --input results.json

# Check results
cat results.json | jq '.total.p50'  # Should see P50 latency
```

---

**Status**: Testing guide complete ✅  
**Next**: Run all tests and measure baseline latency  

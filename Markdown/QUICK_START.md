# Quick Start Guide (30 Minutes)

Get the RAG system up and running quickly with this step-by-step guide.

---

## Prerequisites (2 minutes)

**Check System**:
```bash
python --version         # Should be 3.9+
pip --version           # Check pip
nvidia-smi              # Verify GPU (if available)
```

**Clone Repository**:
```bash
cd ~
git clone https://github.com/YOUR_USERNAME/hhgoa-rag.git
cd hhgoa-rag
```

---

## Step 1: Setup Environment (5 minutes)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Verify activation (should show "(venv)" prefix)
python --version
```

---

## Step 2: Install Dependencies (10 minutes)

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 (GPU)
# For CPU-only, use: --index-url https://download.pytorch.org/whl/cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install project dependencies
pip install -r requirements.txt
```

---

## Step 3: Configure & Initialize (5 minutes)

```bash
# Create environment file
cat > .env << 'EOF'
# API Keys (get from services)
PINECONE_API_KEY=mock
SARVAM_API_KEY=mock
ANTHROPIC_API_KEY=mock

# Environment
DEBUG=true
LOG_LEVEL=INFO
EOF

# Create config.py
cat > config.py << 'EOF'
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "mock")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "mock")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "mock")

EMBEDDING_MODEL = "ai4bharat/indic-bert-v1"
EMBEDDING_DIMENSION = 384

TOP_K_RETRIEVAL = 10
TOP_K_FINAL = 5

MAX_LATENCY_MS = 200
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

USE_LOCAL_LLM = True
USE_CLAUDE_FALLBACK = True
EOF

# Create test data directory
mkdir -p data

# Create test chunks
cat > data/test_chunks.json << 'EOF'
[
  {
    "doc_id": "doc_001",
    "language": "en",
    "content": "Aadhaar is a 12-digit unique identity number issued to all Indian residents by the Unique Identification Authority of India (UIDAI). It serves as proof of identity and address.",
    "metadata": {"section": "Identity", "confidence": 0.95}
  },
  {
    "doc_id": "doc_002",
    "language": "hi",
    "content": "आधार भारत में सभी निवासियों को जारी किया जाने वाला एक 12 अंकों की अद्वितीय पहचान संख्या है।",
    "metadata": {"section": "Parichay", "confidence": 0.92}
  },
  {
    "doc_id": "doc_003",
    "language": "en",
    "content": "To apply for Aadhaar, you need to visit an Aadhaar enrollment center with proof of residence and identity. The process is free.",
    "metadata": {"section": "Application", "confidence": 0.88}
  },
  {
    "doc_id": "doc_004",
    "language": "ta",
    "content": "ஆதார் என்பது இந்தியாவில் உள்ள அனைத்து குடிமக்களுக்கும் வழங்கப்படும் 12 இலக்கக் தனித்துவமான அடையாள எண்.",
    "metadata": {"section": "Identity", "confidence": 0.85}
  }
]
EOF

echo "✓ Configuration complete"
```

---

## Step 4: Verify Installation (3 minutes)

```bash
# Test Python imports
python << 'EOF'
import sys
sys.path.insert(0, '.')

print("Testing imports...")
import torch
print(f"✓ PyTorch {torch.__version__}")

from sentence_transformers import SentenceTransformer
print("✓ Sentence-Transformers")

import pinecone
print("✓ Pinecone")

import whoosh
print("✓ Whoosh")

import fastapi
print("✓ FastAPI")

import anthropic
print("✓ Anthropic")

print("\n✅ All dependencies verified!")
EOF
```

---

## Step 5: Start the Server (2 minutes)

```bash
# Copy main app to src/
mkdir -p src
cp main_app.py src/main.py

# Start server
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Expected output:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

**Keep this terminal open!**

---

## Step 6: Test the API (3 minutes)

### Open another terminal:

```bash
# Activate venv
source venv/bin/activate

# Test health endpoint
curl http://localhost:8000/health

# Expected: {"status": "ok", "models_loaded": true, "ready": true}
```

### Test query endpoint:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "What is Aadhaar?"}'

# Expected:
# {
#   "query": "What is Aadhaar?",
#   "answer": "Aadhaar is a 12-digit unique identity number...",
#   "retrieved_documents": [...],
#   "confidence": 0.87,
#   "latency_breakdown": {"total": "120.5ms"},
#   "status": "ok"
# }
```

### Open Interactive Docs:

```bash
# In your browser, visit:
http://localhost:8000/docs

# Try queries interactively!
```

---

## Quick Test Queries

Try these queries to test the system:

### English
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "How to apply for Aadhaar?"}'
```

### Hindi
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "आधार के लिए आवेदन कैसे करें?"}'
```

### Tamil
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "ஆதாரை விண்ணப்பிப்பது எப்படி?"}'
```

---

## Measure Latency

```bash
# In a third terminal:
source venv/bin/activate

# Run 10 quick queries and measure latency
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query_text": "What is Aadhaar?"}' \
    | jq '.latency_breakdown'
done

# Expected: Each query <200ms
```

---

## What's Working Now

✅ Embedding service (AI4Bharat)  
✅ BM25 search (Whoosh)  
✅ Pinecone mock (in-memory vector DB)  
✅ Result merging (hybrid retrieval)  
✅ FastAPI endpoints  
✅ Latency tracking  
✅ Interactive API docs  

---

## Next Steps (After Quick Start)

### 1. Read Full Documentation
```bash
cat README.md        # Project overview
cat SETUP.md         # Detailed setup
cat ARCHITECTURE.md  # System design
```

### 2. Add Real API Keys
```bash
# Edit .env with real credentials:
# PINECONE_API_KEY=pk_xxx
# SARVAM_API_KEY=xxx
# ANTHROPIC_API_KEY=sk-ant-xxx
```

### 3. Download Full Dataset
```bash
python scripts/download_dataset.py  # MSMARCO-XI
python scripts/chunk_and_index.py   # Chunk & index
```

### 4. Add Generation & Guardrails
```bash
# Create src/generation_service.py (LLM generation)
# Create src/guardrails.py (grounding checks)
# Create src/stt_service.py (speech-to-text)
```

### 5. Run Full Tests
```bash
pytest tests/ -v                    # All tests
python tests/load_test.py          # Latency benchmark
```

### 6. Deploy to VPS
```bash
# See DEPLOYMENT.md for production setup
```

---

## Troubleshooting

### "Module not found"
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify venv is activated
which python  # Should show venv/bin/python
```

### "CUDA not available"
```bash
# Check GPU
nvidia-smi

# If no output, reinstall PyTorch CPU version
pip install torch torchvision torchaudio
```

### "Connection refused on localhost:8000"
```bash
# Make sure server is running in the other terminal
# Check port isn't in use:
lsof -i :8000
```

### "Whoosh index corrupted"
```bash
# Delete and recreate
rm -rf whoosh_index/
python src/whoosh_service.py  # Recreates index
```

---

## Success Checklist

- [ ] venv activated
- [ ] All dependencies installed
- [ ] .env file created
- [ ] config.py created
- [ ] test data at data/test_chunks.json
- [ ] Server running on localhost:8000
- [ ] Health check passes: curl http://localhost:8000/health
- [ ] Query endpoint works
- [ ] Latency <250ms for test queries
- [ ] API docs open in browser: http://localhost:8000/docs

---

## Performance Tips

### Speed Up Model Loading
```bash
# Set environment variable to use GPU
export CUDA_VISIBLE_DEVICES=0
```

### Reduce Memory Usage
```bash
# In config.py, use CPU:
# DEVICE = "cpu"
```

### Check Latency
```bash
# Terminal command to monitor latencies
watch -n 1 'curl -s http://localhost:8000/metrics | jq ".latency_metrics.total"'
```

---

## What to Do Next

1. **Test more queries** → Verify retrieval & generation working
2. **Check latency breakdown** → Identify bottlenecks
3. **Add real API keys** → Connect to Pinecone, Sarvam, Claude
4. **Download full dataset** → Index MSMARCO-XI
5. **Run benchmarks** → Measure P50/P70/P100 latency
6. **Optimize** → Reduce latency, improve quality
7. **Deploy** → Get live endpoint

---

## Getting Help

### Check Logs
```bash
# Server logs show detailed errors
# Look for ✗ markers in console output
```

### Run Diagnostics
```bash
python -c "from src.embedding_service import EmbeddingService; \
           s = EmbeddingService(); \
           print(f'Dimension: {s.get_dimension()}')"
```

### Read Documentation
- README.md - Overview
- SETUP.md - Installation details
- ARCHITECTURE.md - System design
- IMPLEMENTATION_GUIDE.md - Build guide
- TESTING.md - Testing procedures

---

## You're Ready! 🚀

Your RAG system is now running locally. From here:

1. **Explore** the API at http://localhost:8000/docs
2. **Test** with different queries
3. **Measure** latency and quality
4. **Optimize** for production
5. **Deploy** to your VPS
6. **Submit** to the leaderboard

---

**Next**: Follow the full IMPLEMENTATION_GUIDE.md to add LLM generation and guardrails.

**Questions?** Check README.md or TROUBLESHOOTING.md

**Ready to build?** Open IMPLEMENTATION_GUIDE.md and continue!

---

⏱️ **Total setup time**: ~30 minutes  
✅ **Status**: Quick start complete!  
🎯 **Next milestone**: Full pipeline with generation

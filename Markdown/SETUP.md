# Setup & Installation Guide

This guide walks through every step to get your development environment ready for building the RAG system.

**Estimated time**: 30-45 minutes

---

## Prerequisites

### System Requirements
- **OS**: Ubuntu 24 LTS (recommended for VPS), or macOS, or Windows with WSL2
- **Python**: 3.9+ (3.10+ recommended)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (for local Llama inference)
  - Minimum: NVIDIA T4 (16GB VRAM)
  - Recommended: A40 or better
- **RAM**: 16GB minimum
- **Disk**: 50GB free (for dataset, models, indexes)

### Check GPU (if on VPS)
```bash
nvidia-smi
# Expected output: CUDA Version: 12.x, GPU: A100/A40/V100/T4
```

If no GPU output, install CUDA:
```bash
# Ubuntu 24
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1
```

---

## Step 1: Clone Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/hhgoa-rag.git
cd hhgoa-rag
```

---

## Step 2: Create Python Virtual Environment

### macOS / Linux
```bash
python3.10 -m venv venv
source venv/bin/activate

# Verify activation (should show "(venv)" prefix)
python --version  # Should be 3.10+
```

### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python --version
```

---

## Step 3: Install Base Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch GPU support
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected output:
# PyTorch: 2.0.0+cu121
# CUDA available: True
```

---

## Step 4: Install Project Dependencies

```bash
# Create requirements.txt (see below)
cat > requirements.txt << 'EOF'
# Core
python-dotenv==1.0.0
pydantic==2.0.0

# Web Framework
fastapi==0.104.0
uvicorn==0.24.0
httpx==0.25.0

# NLP & ML
sentence-transformers==2.2.2
torch==2.0.0
transformers==4.34.0
numpy==1.24.3
scipy==1.11.0

# Retrieval
whoosh==2.7.4
pinecone-client==3.0.2

# STT (optional, for testing)
requests==2.31.0

# LLM APIs
anthropic==0.7.0

# Testing
pytest==7.4.0
pytest-asyncio==0.21.0

# Utilities
tqdm==4.66.0
pandas==2.1.0
EOF

pip install -r requirements.txt
```

**Installation will take 10-15 minutes** (downloading PyTorch, Transformers, etc.)

---

## Step 5: Download Pre-trained Models

Models will auto-download on first use, but we can pre-download them to save time:

```bash
python << 'EOF'
from sentence_transformers import SentenceTransformer

# Download AI4Bharat IndicBERT
print("Downloading AI4Bharat IndicBERT...")
model = SentenceTransformer('ai4bharat/indic-bert-v1')
print(f"✓ Downloaded to: {model.get_sentence_embedding_dimension()} dimensions")

# Test it works
test_embedding = model.encode("Test query")
print(f"✓ Test embedding shape: {test_embedding.shape}")
EOF
```

**This will take 5-10 minutes** on first run.

---

## Step 6: Configure Environment Variables

### Create `.env` file
```bash
cat > .env << 'EOF'
# API Keys (get these from services)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west-2-aws
PINECONE_INDEX_NAME=hhgoa-rag

SARVAM_API_KEY=your_sarvam_api_key_here

ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Local LLM
LLAMA_MODEL_PATH=/path/to/llama-2-7b.gguf  # Or use HuggingFace auto-download
USE_LOCAL_LLM=true
USE_CLAUDE_FALLBACK=true

# Retrieval
TOP_K_RETRIEVAL=10
TOP_K_FINAL=5

# Latency
MAX_LATENCY_MS=200

# Environment
DEBUG=false
LOG_LEVEL=INFO
EOF

# Protect API keys
chmod 600 .env
```

### Create `config.py` from template
```bash
cat > config.py << 'EOF'
import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west-2-aws")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hhgoa-rag")

# Sarvam (STT)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# Anthropic (Claude API)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# LLM Strategy
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
USE_CLAUDE_FALLBACK = os.getenv("USE_CLAUDE_FALLBACK", "true").lower() == "true"
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.1")

# Embedding Model
EMBEDDING_MODEL = "ai4bharat/indic-bert-v1"
EMBEDDING_DIMENSION = 384

# Retrieval
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))

# Performance
MAX_LATENCY_MS = int(os.getenv("MAX_LATENCY_MS", "200"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Validation
if not PINECONE_API_KEY and not os.getenv("TESTING"):
    print("⚠️  Warning: PINECONE_API_KEY not set. Set it in .env file.")

if DEBUG:
    print(f"Config loaded: {PINECONE_INDEX_NAME}, Model: {LLAMA_MODEL_PATH}")
EOF
```

---

## Step 7: Get API Keys

### Pinecone (Vector Database)
1. Go to https://www.pinecone.io
2. Sign up → Create account
3. Create new index: `hhgoa-rag` (768 dimensions, cosine similarity)
4. Copy API key → paste in `.env`

**Or use mock Pinecone for testing:**
```bash
# In config.py, set PINECONE_API_KEY = "mock"
# Our code handles mock automatically
```

### Sarvam (Speech-to-Text)
1. Go to https://sarvam.ai
2. Sign up → API docs
3. Get API key
4. Paste in `.env`

**Or use mock STT for testing:**
```bash
# In config.py, set SARVAM_API_KEY = "mock"
```

### Anthropic (Claude API)
1. Go to https://console.anthropic.com
2. Create account
3. Get API key from settings
4. Paste in `.env`

**Or skip Claude for now** (use local Llama only, less quality but fast)

---

## Step 8: Verify Installation

```bash
# Test Python imports
python << 'EOF'
import torch
print(f"✓ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

from sentence_transformers import SentenceTransformer
print(f"✓ Sentence-Transformers loaded")

import pinecone
print(f"✓ Pinecone client ready")

import whoosh
print(f"✓ Whoosh indexed")

import fastapi
print(f"✓ FastAPI ready")

import anthropic
print(f"✓ Anthropic client ready")

print("\n✅ All dependencies installed successfully!")
EOF
```

Expected output:
```
✓ PyTorch 2.0.0+cu121, CUDA: True
✓ Sentence-Transformers loaded
✓ Pinecone client ready
✓ Whoosh indexed
✓ FastAPI ready
✓ Anthropic client ready

✅ All dependencies installed successfully!
```

---

## Step 9: Create Test Data

```bash
mkdir -p data

# Create test chunks for initial testing
cat > data/test_chunks.json << 'EOF'
[
  {
    "doc_id": "doc_001",
    "language": "en",
    "content": "Aadhaar is a 12-digit unique identity number issued to all Indian residents by the Unique Identification Authority of India (UIDAI). It serves as proof of identity and address.",
    "metadata": {
      "section": "Identity",
      "source": "MSMARCO-XI",
      "confidence": 0.95
    }
  },
  {
    "doc_id": "doc_002",
    "language": "hi",
    "content": "आधार भारत में सभी निवासियों को जारी किया जाने वाला एक 12 अंकों की अद्वितीय पहचान संख्या है। यह अद्वितीय पहचान प्राधिकरण (UIDAI) द्वारा जारी किया जाता है।",
    "metadata": {
      "section": "Parichay",
      "source": "MSMARCO-XI",
      "confidence": 0.92
    }
  },
  {
    "doc_id": "doc_003",
    "language": "en",
    "content": "To apply for Aadhaar, you need to visit an Aadhaar enrollment center with proof of residence and identity. The process is free and takes about 15 minutes.",
    "metadata": {
      "section": "Application Process",
      "source": "MSMARCO-XI",
      "confidence": 0.88
    }
  },
  {
    "doc_id": "doc_004",
    "language": "ta",
    "content": "ஆதார் என்பது இந்தியாவில் உள்ள அனைத்து குடிமக்களுக்கும் வழங்கப்படும் 12 இலக்கக் தனித்துவமான அடையாள எண்.",
    "metadata": {
      "section": "Identity",
      "source": "MSMARCO-XI",
      "confidence": 0.85
    }
  }
]
EOF

echo "✓ Test data created at data/test_chunks.json"
```

---

## Step 10: Quick Sanity Check

```bash
# Test the embedding service without the full app
python << 'EOF'
import sys
sys.path.insert(0, '.')

from config import EMBEDDING_MODEL
from sentence_transformers import SentenceTransformer

print(f"Loading model: {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)

query = "What is Aadhaar?"
embedding = model.encode(query)

print(f"✓ Query: {query}")
print(f"✓ Embedding dimension: {len(embedding)}")
print(f"✓ Sample values: {embedding[:5]}")
print("\n✅ Embedding service works!")
EOF
```

---

## Troubleshooting

### "CUDA not available"
```bash
# Check GPU
nvidia-smi

# If no output, install CUDA (see step 1)
# If GPU present but PyTorch doesn't see it:
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "Out of memory" when loading model
```bash
# Reduce batch size or use CPU
# In config.py, set:
# DEVICE = "cpu"  # Will be slower but works on CPU-only machines
```

### "Pinecone connection refused"
```bash
# Verify .env file has correct API key
cat .env | grep PINECONE

# Test connection
python << 'EOF'
import pinecone
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
print("✓ Pinecone connected")
EOF
```

### "Module not found"
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Verify venv is activated
which python  # Should show path to venv/bin/python
```

---

## Next Steps

✅ **Setup complete!** Now proceed to:

1. **Read ARCHITECTURE.md** → Understand system design
2. **Follow IMPLEMENTATION_GUIDE.md** → Build the retrieval pipeline
3. **Run tests** → Validate each component

```bash
# Move to implementation
cat IMPLEMENTATION_GUIDE.md
```

---

## Environment Checklist

- [ ] Python 3.9+ installed
- [ ] NVIDIA CUDA 11.8+ (if using GPU)
- [ ] Virtual environment created & activated
- [ ] All dependencies installed (`pip list | grep -E "torch|transformers|pinecone|fastapi"`)
- [ ] `.env` file created with API keys
- [ ] `config.py` created
- [ ] Test data at `data/test_chunks.json`
- [ ] Embedding model downloaded
- [ ] `nvidia-smi` shows GPU (if on GPU machine)

**Status**: Ready for implementation! ✅

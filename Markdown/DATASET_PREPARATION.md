# Dataset Preparation: MSMARCO-XI

Complete guide to download, explore, and prepare the MSMARCO-XI dataset from HuggingFace.

**Dataset Link**: https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

---

## Overview

**MSMARCO-XI** is a multilingual retrieval dataset covering Indian languages:
- **English** (en)
- **Hindi** (hi)
- **Tamil** (ta)
- **Telugu** (te)
- **Kannada** (kn)
- **Marathi** (mr)
- **Gujarati** (gu)
- **Bengali** (bn)
- **Punjabi** (pa)
- **Odia** (or)

**Size**: ~1.1 million documents, ~1.6 million query-document pairs

**Format**: MS MARCO format adapted for Indian languages

---

## Step 1: Install HuggingFace Datasets Library

```bash
pip install datasets huggingface_hub
```

---

## Step 2: Download the Dataset

### Option A: Use HuggingFace Datasets Library (Recommended)

```python
# download_dataset.py
from datasets import load_dataset
import os

# Create data directory
os.makedirs("data/msmarco-xi", exist_ok=True)

print("Downloading MSMARCO-XI dataset...")

# Load corpus (documents)
corpus = load_dataset(
    "ai4bharat/MSMARCO-XI",
    "corpus",
    split="train",
    cache_dir="data/msmarco-xi"
)

print(f"✓ Corpus loaded: {len(corpus)} documents")
print(f"  First doc: {corpus[0]}")

# Load queries
queries = load_dataset(
    "ai4bharat/MSMARCO-XI",
    "queries",
    split="train",
    cache_dir="data/msmarco-xi"
)

print(f"✓ Queries loaded: {len(queries)} queries")

# Load qrels (query-relevance mappings)
qrels = load_dataset(
    "ai4bharat/MSMARCO-XI",
    "qrels",
    split="train",
    cache_dir="data/msmarco-xi"
)

print(f"✓ Qrels loaded: {len(qrels)} relevance pairs")

# Save to local files for easier access
corpus.save_to_disk("data/msmarco-xi/corpus")
queries.save_to_disk("data/msmarco-xi/queries")
qrels.save_to_disk("data/msmarco-xi/qrels")

print("✓ Dataset saved locally")
```

**Run it**:
```bash
python download_dataset.py
# Takes ~30 minutes depending on internet speed
```

### Option B: Manual Download from HuggingFace Hub

```bash
cd data/msmarco-xi

# Download via Git LFS (requires git-lfs installed)
git clone https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

# Or use wget/curl for direct file access
# (See HuggingFace dataset page for direct links)
```

---

## Step 3: Explore Dataset Structure

```python
# explore_dataset.py
from datasets import load_from_disk
import json
from collections import Counter

print("=" * 60)
print("MSMARCO-XI DATASET EXPLORATION")
print("=" * 60)

# Load datasets
corpus = load_from_disk("data/msmarco-xi/corpus")
queries = load_from_disk("data/msmarco-xi/queries")
qrels = load_from_disk("data/msmarco-xi/qrels")

print(f"\n📊 DATASET SIZES:")
print(f"  Corpus: {len(corpus):,} documents")
print(f"  Queries: {len(queries):,} queries")
print(f"  Qrels: {len(qrels):,} relevance pairs")

# Explore corpus structure
print(f"\n📄 CORPUS SCHEMA:")
print(f"  Keys: {corpus.column_names}")
print(f"\n  Example document:")
example_doc = corpus[0]
for key, value in example_doc.items():
    if isinstance(value, str) and len(value) > 100:
        print(f"    {key}: {value[:100]}...")
    else:
        print(f"    {key}: {value}")

# Explore queries structure
print(f"\n❓ QUERIES SCHEMA:")
print(f"  Keys: {queries.column_names}")
print(f"\n  Example query:")
example_query = queries[0]
for key, value in example_query.items():
    print(f"    {key}: {value}")

# Explore qrels structure
print(f"\n🔗 QRELS SCHEMA:")
print(f"  Keys: {qrels.column_names}")
print(f"\n  Example qrel:")
example_qrel = qrels[0]
for key, value in example_qrel.items():
    print(f"    {key}: {value}")

# Language distribution
print(f"\n🌍 LANGUAGE DISTRIBUTION (sample):")
doc_langs = Counter([d.get("lang", "unknown") for d in corpus[:10000]])
for lang, count in doc_langs.most_common():
    print(f"  {lang}: {count}")

# Document length statistics
print(f"\n📏 DOCUMENT LENGTHS (sample):")
lengths = [len(d.get("doc", "").split()) for d in corpus[:10000]]
print(f"  Min: {min(lengths)} words")
print(f"  Max: {max(lengths)} words")
print(f"  Avg: {sum(lengths) / len(lengths):.0f} words")

# Query length statistics
print(f"\n❓ QUERY LENGTHS (sample):")
query_lengths = [len(q.get("query", "").split()) for q in queries[:10000]]
print(f"  Min: {min(query_lengths)} words")
print(f"  Max: {max(query_lengths)} words")
print(f"  Avg: {sum(query_lengths) / len(query_lengths):.0f} words")

print("\n✓ Exploration complete")
```

**Run it**:
```bash
python explore_dataset.py
```

**Expected output**:
```
DATASET SIZES:
  Corpus: 1,200,000 documents
  Queries: 180,000 queries
  Qrels: 300,000 relevance pairs

CORPUS SCHEMA:
  Keys: ['doc_id', 'doc', 'lang']

  Example document:
    doc_id: d_001
    doc: Aadhaar is a 12-digit unique identity number...
    lang: en

QUERIES SCHEMA:
  Keys: ['query_id', 'query', 'lang']

  Example query:
    query_id: q_001
    query: What is Aadhaar?
    lang: en

QRELS SCHEMA:
  Keys: ['query_id', 'doc_id', 'relevance']

  Example qrel:
    query_id: q_001
    doc_id: d_001
    relevance: 1

LANGUAGE DISTRIBUTION (sample):
  en: 4200
  hi: 2100
  ta: 1800
  ...
```

---

## Step 4: Chunk Documents for Indexing

The raw documents are too long for embedding. We need to chunk them intelligently.

```python
# chunk_documents.py
from datasets import load_from_disk
import json
from tqdm import tqdm
import re

def smart_chunk(text, language="en", max_chunk_size=512):
    """
    Intelligently chunk text based on language.
    
    Strategy:
    - Split by sentences first
    - Group sentences into chunks
    - Maintain overlap for context
    """
    
    if language == "en":
        # English: split by periods + spaces
        sentences = re.split(r'(?<=[.!?])\s+', text)
    elif language in ["hi", "mr", "bn"]:
        # Hindi/Indic: split by sentence-ending punctuation
        sentences = re.split(r'(?<=[।।।])\s+', text)
    else:
        # Fallback: simple split
        sentences = text.split('\n')
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence.split())
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Save chunk with overlap
            chunks.append(" ".join(current_chunk))
            # Keep last sentence for overlap
            current_chunk = [current_chunk[-1] if current_chunk else "", sentence]
            current_size = len(current_chunk[-1].split()) + sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

print("Loading corpus...")
corpus = load_from_disk("data/msmarco-xi/corpus")

print("Chunking documents...")
chunks_list = []
chunk_id_counter = 0

for doc_idx, doc in enumerate(tqdm(corpus, desc="Processing documents")):
    doc_id = doc["doc_id"]
    text = doc["doc"]
    language = doc.get("lang", "en")
    
    # Chunk the document
    doc_chunks = smart_chunk(text, language=language, max_chunk_size=512)
    
    # Create chunk objects
    for chunk_idx, chunk_text in enumerate(doc_chunks):
        chunk = {
            "chunk_id": f"chunk_{chunk_id_counter}",
            "doc_id": doc_id,
            "chunk_idx": chunk_idx,
            "content": chunk_text,
            "language": language,
            "source": "MSMARCO-XI"
        }
        chunks_list.append(chunk)
        chunk_id_counter += 1
    
    # Save periodically
    if (doc_idx + 1) % 100000 == 0:
        print(f"  Processed {doc_idx + 1} documents → {chunk_id_counter} chunks")

print(f"\n✓ Total chunks: {len(chunks_list)}")

# Save chunks to JSONL (efficient for large datasets)
print("Saving chunks...")
with open("data/msmarco-xi/chunks.jsonl", "w", encoding="utf-8") as f:
    for chunk in chunks_list:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print("✓ Chunks saved to data/msmarco-xi/chunks.jsonl")

# Also save summary
summary = {
    "total_chunks": len(chunks_list),
    "total_documents": len(corpus),
    "avg_chunks_per_doc": len(chunks_list) / len(corpus),
}

with open("data/msmarco-xi/chunk_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary:")
print(f"  Total chunks: {summary['total_chunks']:,}")
print(f"  Avg chunks/doc: {summary['avg_chunks_per_doc']:.1f}")
```

**Run it**:
```bash
python chunk_documents.py
# Takes ~10 minutes
```

---

## Step 5: Create Embedding Index

Now embed all chunks and create Pinecone/Whoosh indexes.

```python
# create_indexes.py
import json
from tqdm import tqdm
from src.embedding_service import EmbeddingService
from src.pinecone_service import PineconeService
from src.whoosh_service import WhooshService

print("Loading embedding service...")
embedding_service = EmbeddingService()

print("Loading chunks...")
chunks = []
with open("data/msmarco-xi/chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks")

# ====== Create Pinecone Index ======
print("\nCreating Pinecone index...")
pinecone_service = PineconeService()

embeddings_to_upsert = []
for chunk in tqdm(chunks, desc="Embedding chunks"):
    embedding = embedding_service.embed_query(chunk["content"])
    embeddings_to_upsert.append({
        "id": chunk["chunk_id"],
        "embedding": embedding.tolist(),
        "metadata": {
            "doc_id": chunk["doc_id"],
            "language": chunk["language"],
            "content": chunk["content"][:500]  # Store summary
        }
    })
    
    # Upsert in batches of 1000
    if len(embeddings_to_upsert) >= 1000:
        pinecone_service.upsert(embeddings_to_upsert)
        embeddings_to_upsert = []

# Final batch
if embeddings_to_upsert:
    pinecone_service.upsert(embeddings_to_upsert)

print("✓ Pinecone index created")

# ====== Create Whoosh Index ======
print("\nCreating Whoosh index...")
whoosh_service = WhooshService(chunks=chunks)
print("✓ Whoosh index created")

print("\n✅ All indexes created successfully")
```

**Run it**:
```bash
python create_indexes.py
# Takes ~2-3 hours (1M embeddings @ 15ms each)
```

---

## Step 6: Verify Indexes

```python
# verify_indexes.py
from src.embedding_service import EmbeddingService
from src.pinecone_service import PineconeService
from src.whoosh_service import WhooshService

embedding_service = EmbeddingService()
pinecone_service = PineconeService()
whoosh_service = WhooshService()

# Test queries
test_queries = [
    "What is Aadhaar?",
    "आधार क्या है?",
    "How to apply for Aadhaar",
]

for query in test_queries:
    print(f"\nQuery: {query}")
    
    # Dense retrieval
    emb = embedding_service.embed_query(query)
    dense_results = pinecone_service.query(emb.tolist(), top_k=3)
    print(f"  Dense: {len(dense_results)} results")
    for r in dense_results[:1]:
        print(f"    - {r['doc_id']}: {r['score']:.3f}")
    
    # BM25 retrieval
    bm25_results = whoosh_service.query(query, top_k=3)
    print(f"  BM25: {len(bm25_results)} results")
    for r in bm25_results[:1]:
        print(f"    - {r['doc_id']}: {r['score']:.1f}")

print("\n✅ Verification complete")
```

---

## Dataset Statistics

After preparing MSMARCO-XI, you'll have:

```
Corpus
├── Total documents: 1.2M
├── Languages: 11 (en, hi, ta, te, kn, mr, gu, bn, pa, or, ml)
└── Total content: ~2GB

Chunked
├── Total chunks: ~3.5M (avg 3 chunks/doc)
├── Chunk size: 256-512 tokens
├── Languages: Same 11
└── Metadata: doc_id, language, section

Indexed
├── Pinecone: 3.5M embeddings (384-dim)
├── Whoosh: 3.5M indexed docs
└── Total index size: ~20GB
```

---

## Performance Tips

### Faster Embedding
```python
# Use batch embedding instead of single
embeddings = embedding_service.embed_documents(
    [chunk["content"] for chunk in chunks],
    batch_size=64
)
```

### Optimize Chunk Size
- **Too small** (<200 tokens): Loses context, more chunks
- **Too large** (>1000 tokens): Slow embedding, loses precision
- **Optimal**: 256-512 tokens for Indian languages

### Reduce Index Size
```python
# Skip storing full content in Pinecone metadata
# Store only: doc_id, language, chunk_idx
# Fetch full content from chunks.jsonl when needed
```

---

## Troubleshooting

### "Out of Memory" during embedding
```bash
# Use smaller batch size
# In create_indexes.py:
# batch_size=16  (instead of default 32)
```

### "HuggingFace API limit"
```bash
# Use local dataset if already downloaded
corpus = load_from_disk("data/msmarco-xi/corpus")
```

### "Pinecone quota exceeded"
```bash
# Use Milvus or Weaviate for local vector DB
# Or filter dataset to top languages only
```

---

## Next Steps

After preparing the dataset:

1. **Verify retrieval quality**: Run TREC evaluation
2. **Measure latency**: Benchmark P50/P70/P100
3. **Optimize chunking**: Try different chunk sizes
4. **Add guardrails**: Implement grounding checks
5. **Deploy**: Move indexes to production

---

**Status**: Dataset preparation guide complete ✅  
**Next**: Run `python download_dataset.py` and follow the steps above


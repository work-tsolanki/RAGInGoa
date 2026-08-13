"""
Main FastAPI application for Voice-Enabled RAG System
File: src/main.py

This orchestrates all services: STT, Embedding, Retrieval, LLM Generation, Guardrails
"""

import time
import json
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, '.')

from config import (
    DEBUG, LOG_LEVEL, MAX_LATENCY_MS, TOP_K_FINAL,
    USE_LOCAL_LLM, USE_CLAUDE_FALLBACK
)
from src.embedding_service import EmbeddingService
from src.whoosh_service import WhooshService
from src.pinecone_service import PineconeService
from src.retrieval import merge_and_rank
from src.generation_service import GenerationService
from src.guardrails import Guardrails
from src.latency_tracker import get_tracker

# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="HH Goa 2026 Voice-Enabled RAG",
    description="Multilingual RAG system with speech-to-text, retrieval, and generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Data Models
# ============================================================================

class QueryRequest(BaseModel):
    query_text: str
    language: Optional[str] = "en"
    top_k: Optional[int] = TOP_K_FINAL

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[dict]
    confidence: float
    latency_breakdown: dict
    status: str = "ok"

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    ready: bool

# ============================================================================
# Global Services
# ============================================================================

embedding_service = None
whoosh_service = None
pinecone_service = None
generation_service = None
guardrails = None

def initialize_services():
    """Initialize all services on startup."""
    global embedding_service, whoosh_service, pinecone_service, generation_service, guardrails
    
    print("[main.py] Initializing services...")
    
    try:
        # Embedding
        print("  → Loading embedding model...")
        embedding_service = EmbeddingService()
        
        # Whoosh (load test data)
        print("  → Loading Whoosh index...")
        with open("data/test_chunks.json", encoding="utf-8") as f:
            chunks = json.load(f)
        whoosh_service = WhooshService(chunks=chunks)
        
        # Pinecone (with mock fallback)
        print("  → Initializing Pinecone...")
        pinecone_service = PineconeService(use_mock=True)
        
        # Index chunks in Pinecone
        embeddings_to_upsert = []
        for chunk in chunks:
            emb = embedding_service.embed_query(chunk["content"])
            metadata = dict(chunk.get("metadata", {}))
            metadata["content"] = chunk["content"]
            embeddings_to_upsert.append({
                "id": chunk["doc_id"],
                "embedding": emb.tolist(),
                "metadata": metadata
            })
        pinecone_service.upsert(embeddings_to_upsert)
        
        # Generation
        print("  → Loading LLM generation service...")
        generation_service = GenerationService(use_local=USE_LOCAL_LLM)
        
        # Guardrails
        print("  → Loading guardrails...")
        guardrails = Guardrails()
        
        print("✓ All services initialized successfully")
        return True
    
    except Exception as e:
        print(f"✗ Failed to initialize services: {e}")
        return False

# ============================================================================
# Routes
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Called when app starts."""
    success = initialize_services()
    if not success:
        raise RuntimeError("Failed to initialize services")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        models_loaded=embedding_service is not None,
        ready=all([
            embedding_service,
            whoosh_service,
            pinecone_service,
            generation_service,
            guardrails
        ])
    )

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Process a text query through the full RAG pipeline.
    
    Args:
        request: QueryRequest with query_text, language, top_k
    
    Returns:
        QueryResponse with answer, retrieved docs, confidence, latency
    """
    pipeline_start = time.time()
    latency_breakdown = {}
    
    try:
        # ====== Step 1: Input Validation ======
        if not request.query_text or len(request.query_text.strip()) == 0:
            raise ValueError("Query cannot be empty")
        
        # ====== Step 2: Embed Query ======
        start = time.time()
        query_embedding = embedding_service.embed_query(request.query_text)
        latency_breakdown["embedding"] = (time.time() - start) * 1000
        
        # ====== Step 3: Parallel Retrieval ======
        start = time.time()
        dense_results = pinecone_service.query(
            query_embedding.tolist(),
            top_k=10
        )
        bm25_results = whoosh_service.query(
            request.query_text,
            top_k=10
        )
        latency_breakdown["retrieval"] = (time.time() - start) * 1000
        
        # ====== Step 4: Merge Results ======
        start = time.time()
        retrieved_docs = merge_and_rank(
            dense_results,
            bm25_results,
            top_k=request.top_k
        )
        latency_breakdown["merge"] = (time.time() - start) * 1000
        
        # ====== Step 5: Check Elapsed Time ======
        elapsed_ms = (time.time() - pipeline_start) * 1000
        if DEBUG:
            print(f"[pipeline] Elapsed: {elapsed_ms:.0f}ms / {MAX_LATENCY_MS}ms")
        
        # ====== Step 6: Generate Answer ======
        start = time.time()
        
        # Prepare context
        context_docs = [doc["content"] for doc in retrieved_docs]
        answer = generation_service.generate(
            query=request.query_text,
            context=context_docs,
            use_fast_path=(elapsed_ms < 100)  # If we have time budget, use local LLM
        )
        
        latency_breakdown["generation"] = (time.time() - start) * 1000
        
        # ====== Step 7: Grounding Check ======
        start = time.time()
        grounding_score = guardrails.check_grounding(
            answer=answer,
            retrieved_docs=context_docs
        )
        latency_breakdown["grounding"] = (time.time() - start) * 1000
        
        # ====== Step 8: Validate Answer ======
        if not guardrails.validate_answer(answer):
            answer = "I could not find a clear answer to your question. Please try rephrasing."
            grounding_score = 0.5
        
        # ====== Step 9: Format Response ======
        total_latency = (time.time() - pipeline_start) * 1000
        latency_breakdown["total"] = total_latency
        
        return QueryResponse(
            query=request.query_text,
            answer=answer,
            retrieved_documents=[
                {
                    "doc_id": doc["doc_id"],
                    "content": doc["content"][:200],  # Truncate for response
                    "score": float(doc["final_score"])
                }
                for doc in retrieved_docs
            ],
            confidence=float(grounding_score),
            latency_breakdown={k: f"{v:.1f}ms" for k, v in latency_breakdown.items()},
            status="ok"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if DEBUG:
            print(f"✗ Query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/query_audio")
async def query_audio_endpoint(
    audio: UploadFile = File(...),
    language: str = "en"
):
    """
    Process audio query through full pipeline.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        language: Audio language (en, hi, ta, etc.)
    
    Returns:
        Same as /query endpoint
    """
    pipeline_start = time.time()
    
    try:
        # Read audio file
        audio_bytes = await audio.read()
        
        # ====== STT (Mock for now) ======
        # In production: call Sarvam API
        query_text = f"[STT output for {language}]"  # Mock
        latency_breakdown = {"stt": 45}
        
        # ====== Reuse query endpoint ======
        request = QueryRequest(query_text=query_text, language=language)
        return await query_endpoint(request)
    
    except Exception as e:
        if DEBUG:
            print(f"✗ Audio query failed: {e}")
        raise HTTPException(status_code=500, detail="Audio processing failed")

@app.get("/metrics")
async def get_metrics():
    """Get latency metrics (P50, P70, P100)."""
    tracker = get_tracker()
    stats = tracker.get_stats()
    
    return {
        "latency_metrics": stats,
        "timestamp": time.time()
    }

# ============================================================================
# Development Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   HH Goa 2026: Voice-Enabled RAG System                    ║
    ║   Server starting...                                        ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )

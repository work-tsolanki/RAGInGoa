"""
Main FastAPI application for Voice-Enabled RAG System
File: src/main.py

This orchestrates all services: STT, Embedding, Retrieval, LLM Generation, Guardrails
"""

import asyncio
import json
import secrets
import time
from pathlib import Path
from typing import Optional, List
from fastapi import (
    FastAPI, UploadFile, File, HTTPException, Header, Depends,
    WebSocket, WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, '.')

from config import (
    DEBUG, LOG_LEVEL, MAX_LATENCY_MS, TOP_K_FINAL,
    USE_LOCAL_LLM, USE_CLAUDE_FALLBACK, RAG_API_KEY
)
from src.embedding_service import EmbeddingService
from src.whoosh_service import WhooshService
from src.chroma_service import ChromaService
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
# allow_credentials=False: auth is via X-API-Key header, not cookies, and the
# combination of allow_origins=["*"] with allow_credentials=True is both
# insecure and rejected by browsers anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


DASHBOARD_HTML_PATH = Path(__file__).parent / "dashboard" / "index.html"


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard/", response_class=HTMLResponse)
async def dashboard():
    """
    Serves the local dashboard with the server's own RAG_API_KEY injected at
    request time, so it never has to be typed or stored in the browser
    (localStorage, cookies, etc). Deliberately unauthenticated - but that's
    only safe because uvicorn.run() below binds to 127.0.0.1 only. On any
    host reachable off-machine, this route would hand the key to anyone
    who asks; it would need to gate on verify_api_key (accepting the key via
    a query param for that one bootstrap request) before being exposed like
    that again.
    """
    html = DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
    # Escaped so a key containing "</script>" (not possible with the current
    # generated key format, but not guaranteed) can't break out of the
    # injected <script> block.
    safe_key = json.dumps(RAG_API_KEY or "")[1:-1].replace("</", "<\\/")
    html = html.replace("__RAG_API_KEY_INJECTED__", safe_key)
    return HTMLResponse(html)


if not RAG_API_KEY:
    print("WARNING: RAG_API_KEY is not set - /query and /query_audio are UNAUTHENTICATED. "
          "Set RAG_API_KEY in .env before exposing this server publicly.")


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Require a matching X-API-Key header when RAG_API_KEY is configured.
    If RAG_API_KEY isn't set (e.g. local dev), auth is skipped - see the
    startup warning above, which makes this fail-open behavior visible
    rather than silent."""
    if RAG_API_KEY and not secrets.compare_digest(x_api_key or "", RAG_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

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
chroma_service = None
generation_service = None
guardrails = None

def initialize_services():
    """Initialize all services on startup."""
    global embedding_service, whoosh_service, chroma_service, generation_service, guardrails
    
    print("[main.py] Initializing services...")
    
    try:
        # Embedding
        print("  → Loading embedding model...")
        embedding_service = EmbeddingService()

        # Whoosh: open the pre-built full-scale index (743k+ MSMARCO-XI passages,
        # built offline by scripts/chunk_and_index.py - do not rebuild on every startup)
        print("  → Opening full-scale Whoosh index...")
        whoosh_service = WhooshService(index_dir="whoosh_index_full")

        # Chroma: open the pre-built full-scale persistent collection
        print("  → Opening full-scale Chroma collection...")
        chroma_service = ChromaService(collection_name="hhgoa_rag_full")

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
            chroma_service,
            generation_service,
            guardrails
        ])
    )

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
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
        dense_results = chroma_service.query(
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

@app.websocket("/ws/query")
async def ws_query_endpoint(websocket: WebSocket):
    """
    Streaming version of /query for the local dashboard: pushes one JSON event
    per pipeline stage (start/done + timing) instead of waiting for the full
    response, so the UI can show live progress.

    Browsers' native WebSocket API can't set custom headers, so auth happens
    after the handshake: the client's first message must be
    {"type": "auth", "api_key": ...}. A query-string api_key was considered
    and rejected - it would land in uvicorn's access log and browser history
    in cleartext.
    """
    await websocket.accept()

    if RAG_API_KEY:
        try:
            auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        except Exception:
            await websocket.close(code=4401, reason="Auth message required")
            return
        submitted_key = auth_msg.get("api_key") if isinstance(auth_msg, dict) else None
        if not secrets.compare_digest(submitted_key or "", RAG_API_KEY):
            await websocket.close(code=4401, reason="Invalid API key")
            return

    await websocket.send_json({"stage": "auth", "status": "ok"})

    try:
        while True:
            request = await websocket.receive_json()
            query_text = (request.get("query_text") or "").strip()
            top_k = request.get("top_k") or TOP_K_FINAL

            if not query_text:
                await websocket.send_json({"stage": "error", "message": "Query cannot be empty"})
                continue

            pipeline_start = time.time()
            latency_breakdown = {}

            try:
                await websocket.send_json({"stage": "embedding", "status": "start"})
                start = time.time()
                query_embedding = await asyncio.to_thread(embedding_service.embed_query, query_text)
                latency_breakdown["embedding"] = (time.time() - start) * 1000
                await websocket.send_json({
                    "stage": "embedding", "status": "done",
                    "latency_ms": round(latency_breakdown["embedding"], 1),
                })

                await websocket.send_json({"stage": "retrieval", "status": "start"})
                start = time.time()
                dense_results = await asyncio.to_thread(chroma_service.query, query_embedding.tolist(), top_k=10)
                bm25_results = await asyncio.to_thread(whoosh_service.query, query_text, top_k=10)
                latency_breakdown["retrieval"] = (time.time() - start) * 1000
                await websocket.send_json({
                    "stage": "retrieval", "status": "done",
                    "latency_ms": round(latency_breakdown["retrieval"], 1),
                    "dense_count": len(dense_results), "bm25_count": len(bm25_results),
                })

                start = time.time()
                retrieved_docs = merge_and_rank(dense_results, bm25_results, top_k=top_k)
                latency_breakdown["merge"] = (time.time() - start) * 1000
                await websocket.send_json({
                    "stage": "merge", "status": "done",
                    "latency_ms": round(latency_breakdown["merge"], 1),
                    "documents": [
                        {"doc_id": doc["doc_id"], "content": doc["content"][:200], "score": float(doc["final_score"])}
                        for doc in retrieved_docs
                    ],
                })

                context_docs = [doc["content"] for doc in retrieved_docs]
                await websocket.send_json({"stage": "generation", "status": "start"})
                start = time.time()
                answer = await asyncio.to_thread(generation_service.generate, query_text, context_docs)
                latency_breakdown["generation"] = (time.time() - start) * 1000
                await websocket.send_json({
                    "stage": "generation", "status": "done",
                    "latency_ms": round(latency_breakdown["generation"], 1),
                    "answer": answer,
                })

                await websocket.send_json({"stage": "grounding", "status": "start"})
                start = time.time()
                grounding_score = await asyncio.to_thread(guardrails.check_grounding, answer, context_docs)
                latency_breakdown["grounding"] = (time.time() - start) * 1000
                await websocket.send_json({
                    "stage": "grounding", "status": "done",
                    "latency_ms": round(latency_breakdown["grounding"], 1),
                    "score": grounding_score,
                })

                if not guardrails.validate_answer(answer):
                    answer = "I could not find a clear answer to your question. Please try rephrasing."
                    grounding_score = 0.5

                latency_breakdown["total"] = (time.time() - pipeline_start) * 1000
                await websocket.send_json({
                    "stage": "final",
                    "query": query_text,
                    "answer": answer,
                    "confidence": float(grounding_score),
                    "retrieved_documents": [
                        {"doc_id": doc["doc_id"], "content": doc["content"][:200], "score": float(doc["final_score"])}
                        for doc in retrieved_docs
                    ],
                    "latency_breakdown": {k: round(v, 1) for k, v in latency_breakdown.items()},
                })
            except Exception as e:
                if DEBUG:
                    print(f"[ws_query] Pipeline error: {e}")
                await websocket.send_json({"stage": "error", "message": "Internal server error"})
    except WebSocketDisconnect:
        pass


@app.post("/query_audio", dependencies=[Depends(verify_api_key)])
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
        # Loopback-only: /dashboard hands the RAG_API_KEY to whoever loads
        # it, unauthenticated, so it must not be reachable off this machine.
        # This is the local-only deployment target - if remote/LAN access is
        # ever needed again, /dashboard has to gate on verify_api_key first.
        host="127.0.0.1",
        port=8000,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )

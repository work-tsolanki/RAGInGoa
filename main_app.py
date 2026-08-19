"""
Main FastAPI application for Voice-Enabled RAG System
File: src/main.py

This orchestrates all services: STT, Embedding, Retrieval, LLM Generation, Guardrails
"""

import asyncio
import base64
import secrets
import time
from pathlib import Path

import httpx
from typing import Optional, List
from fastapi import (
    FastAPI, UploadFile, File, HTTPException, Header, Depends, Request,
    WebSocket, WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, '.')

from config import (
    DEBUG, LOG_LEVEL, MAX_LATENCY_MS, TOP_K_FINAL,
    USE_LOCAL_LLM, USE_CLAUDE_FALLBACK, RAG_API_KEY,
    ANSWER_CACHE_MAX_SIZE, ANSWER_CACHE_TTL_SECONDS, ANSWER_CACHE_MIN_GROUNDING,
    SEMANTIC_CACHE_MAX_SIZE, SEMANTIC_CACHE_TTL_SECONDS, SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
    CHROMA_PERSIST_DIR, BM25S_INDEX_DIR,
)
from src.embedding_service import EmbeddingService
from src.bm25s_service import Bm25sService
from src.chroma_service import ChromaService
from src.retrieval import merge_and_rank
from src.generation_service import GenerationService, clean_answer_boilerplate
from src.guardrails import Guardrails, build_low_grounding_response, build_off_topic_response, build_unsafe_response, check_unsafe
from src.stt_service import SttService, RealtimeSttSession, sarvam_code_to_language
from src.answer_cache import AnswerCache, make_cache_key
from src.semantic_cache import SemanticCache
from src.latency_tracker import get_tracker
from src.rate_limiter import RateLimiter

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
    Serves the static dashboard shell - no server-side templating, no key
    injection. Previously injected RAG_API_KEY into the page here, which
    was only safe because uvicorn.run() below bound to 127.0.0.1
    exclusively (nobody who couldn't already reach the API could reach the
    dashboard either). That assumption breaks the moment this deploys
    publicly - anyone who loads the page could read the key out of the
    source regardless of how it got there. Fixed by removing the injection
    entirely: the dashboard now prompts for the key client-side and holds
    it in a JS variable only (see dashboard/index.html) - never rendered
    into the page, never persisted to localStorage/cookies.
    """
    html = DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
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
# Public demo path (no API key) - /demo/query and /ws/demo below
# ============================================================================
# The dashboard is meant to be usable by a judge with zero setup - requiring
# RAG_API_KEY there defeats that, and the key was already deliberately
# removed from server-rendered HTML earlier (see dashboard() above) so it
# can't just be re-exposed to satisfy this. Instead: a separate, keyless
# path protected by a per-IP rate limit instead of a credential. Generation
# config, retrieval, and the off-topic/unsafe guardrail are identical to the
# authenticated path - this only changes how access is gated, not what the
# system does once a request is let through.
DEMO_RATE_LIMIT_MAX_REQUESTS = 10
DEMO_RATE_LIMIT_WINDOW_SECONDS = 60
demo_rate_limiter = RateLimiter(
    max_requests=DEMO_RATE_LIMIT_MAX_REQUESTS,
    window_seconds=DEMO_RATE_LIMIT_WINDOW_SECONDS,
)
DEMO_RATE_LIMIT_MESSAGE = (
    f"You've hit the demo rate limit ({DEMO_RATE_LIMIT_MAX_REQUESTS} queries per "
    f"{DEMO_RATE_LIMIT_WINDOW_SECONDS}s). Please wait a moment and try again."
)


def _client_ip(request_or_websocket) -> str:
    """Fly-Client-IP is set by Fly's edge proxy and confirmed (via
    /debug/client_ip against the live deployment, key-gated) to be
    overwritten on the way in regardless of what value a caller sends -
    4 back-to-back requests spoofing different values all arrived at the
    app as the same real source IP. request.client.host is NOT a usable
    substitute here: Fly's proxy terminates the public connection and
    forwards over its private 6PN network, so client.host is the proxy's
    own internal address (e.g. 172.16.x.x) - identical for every request
    regardless of caller, which would make the rate limit below a single
    shared global budget instead of a per-caller one. (An earlier version
    of this function used client.host after a rate-limit test appeared to
    show the header being spoofed; that test result turned out to be a
    false positive from the limiter's window naturally expiring between
    test batches, not an actual bypass - re-tested back-to-back with no
    delay before reverting to this.)
    .client.host is still the fallback for local dev, where there's no
    Fly edge in front of the app at all and this header won't be present."""
    fly_client_ip = request_or_websocket.headers.get("fly-client-ip")
    if fly_client_ip:
        return fly_client_ip
    client = request_or_websocket.client
    return client.host if client else "unknown"


def verify_demo_rate_limit(http_request: Request):
    if not demo_rate_limiter.allow(_client_ip(http_request)):
        raise HTTPException(status_code=429, detail=DEMO_RATE_LIMIT_MESSAGE)

# ============================================================================
# Data Models
# ============================================================================

# Bounds the public, rate-limited-but-not-size-limited /demo/query and
# /ws/demo paths - a real spoken/typed question is well under this; without
# it an unauth'd caller could repeatedly send oversized payloads within the
# rate-limit window. Enforced twice: Pydantic validates it on the REST path
# (QueryRequest below), but /ws/demo and /ws/query parse raw JSON and never
# go through that model, so _ws_handler checks it explicitly too.
MAX_QUERY_TEXT_LENGTH = 500


class QueryRequest(BaseModel):
    query_text: str = Field(..., max_length=MAX_QUERY_TEXT_LENGTH)
    language: Optional[str] = "en"
    top_k: Optional[int] = TOP_K_FINAL

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[dict]
    confidence: float
    latency_breakdown: dict
    status: str = "ok"
    cache_hit: bool = False
    backend: Optional[str] = None
    cache_type: Optional[str] = None  # "semantic" | "literal" | None (full miss)
    semantic_similarity: Optional[float] = None
    hedged: bool = False  # True when the answer was replaced with a low-grounding hedge message
    declined: bool = False  # True when the query was rejected pre-retrieval (off-topic or unsafe)
    decline_reason: Optional[str] = None  # "off_topic" | "unsafe" | None

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    ready: bool

# ============================================================================
# Global Services
# ============================================================================

embedding_service = None
bm25_service = None
chroma_service = None
generation_service = None
guardrails = None
# No I/O/model load needed, so this doesn't need to wait for
# initialize_services() like the model-backed globals above.
answer_cache = AnswerCache(max_size=ANSWER_CACHE_MAX_SIZE, ttl_seconds=ANSWER_CACHE_TTL_SECONDS)
semantic_cache = SemanticCache(
    max_size=SEMANTIC_CACHE_MAX_SIZE,
    ttl_seconds=SEMANTIC_CACHE_TTL_SECONDS,
    similarity_threshold=SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
)
stt_service = SttService()

def initialize_services():
    """Initialize all services on startup."""
    global embedding_service, bm25_service, chroma_service, generation_service, guardrails
    
    print("[main.py] Initializing services...")
    
    try:
        # Embedding
        print("  → Loading embedding model...")
        embedding_service = EmbeddingService()

        # bm25s: open the pre-built full-scale index (743k+ MSMARCO-XI passages,
        # built offline by scripts/build_bm25s_index.py - do not rebuild on every startup)
        print("  → Opening full-scale bm25s index...")
        bm25_service = Bm25sService(index_dir=BM25S_INDEX_DIR)

        # Chroma: open the pre-built full-scale persistent collection
        print("  → Opening full-scale Chroma collection...")
        chroma_service = ChromaService(collection_name="hhgoa_rag_full", persist_dir=CHROMA_PERSIST_DIR)

        # Generation
        print("  → Loading LLM generation service...")
        generation_service = GenerationService(use_local=USE_LOCAL_LLM)
        
        # Guardrails
        print("  → Loading guardrails...")
        guardrails = Guardrails(embedding_service=embedding_service)

        # Warm-up: first call into each backend pays a one-time cache/JIT
        # cost (CUDA context, HNSW page-in, etc.) that otherwise lands on
        # whichever user sends the first real request. Eat it here instead.
        print("  → Warming up retrieval path...")
        warm_embedding = embedding_service.embed_query("warmup query")
        chroma_service.query(warm_embedding.tolist(), top_k=10)
        bm25_service.query("warmup query", top_k=10)

        print("  → Warming up generation + guardrails...")
        warm_context = ["Warmup context passage for CUDA graph and KV-cache priming."]
        warm_answer = generation_service.generate("warmup query", warm_context)
        guardrails.check_grounding(warm_answer, warm_context)

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
            bm25_service,
            chroma_service,
            generation_service,
            guardrails
        ])
    )


@app.get("/debug/client_ip", dependencies=[Depends(verify_api_key)])
async def debug_client_ip(request: Request):
    """Temporary - added to verify what the rate limiter actually sees
    (request.client.host) against reality on the live deployment, after a
    header-trust assumption (Fly-Client-IP) turned out to be wrong. Remove
    once that's confirmed. Key-gated (an unauthenticated version of this
    briefly shipped and was caught by security review before being relied
    on for anything - raw request headers can reveal infra details with no
    legitimate reason to be public)."""
    return {
        "client_host": request.client.host if request.client else None,
        "resolved_client_ip": _client_ip(request),
    }

@app.get("/debug/cache_stats", dependencies=[Depends(verify_api_key)])
async def cache_stats():
    """Cache hit/miss stats, broken out per layer so you can see which one
    is actually doing the work in real traffic - literal (exact repeat) vs
    semantic (fuzzy match, checked first, before retrieval even runs)."""
    return {
        "literal": answer_cache.stats(),
        "semantic": semantic_cache.stats(),
    }

async def _handle_query(request: QueryRequest) -> QueryResponse:
    """
    Process a text query through the full RAG pipeline. Shared by both
    /query (key-gated) and /demo/query (rate-limited, no key) - identical
    pipeline either way, only the access gate differs (see verify_api_key
    vs verify_demo_rate_limit).

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

        # ====== Step 1b: Unsafe-Input Check ======
        # Pure text pattern match, no embedding/retrieval/generation cost -
        # this is "shouldn't have been processed as a real query at all,"
        # a different failure mode than an ungrounded answer.
        if check_unsafe(request.query_text):
            total_latency = (time.time() - pipeline_start) * 1000
            latency_breakdown["total"] = total_latency
            return QueryResponse(
                query=request.query_text,
                answer=build_unsafe_response(request.language),
                retrieved_documents=[],
                confidence=0.0,
                latency_breakdown={k: f"{v:.1f}ms" for k, v in latency_breakdown.items()},
                status="declined",
                declined=True,
                decline_reason="unsafe",
            )

        # ====== Step 2: Embed Query ======
        start = time.time()
        query_embedding = embedding_service.embed_query(request.query_text)
        latency_breakdown["embedding"] = (time.time() - start) * 1000

        # ====== Step 2a: Off-Topic Check ======
        # Before the semantic/literal cache lookup, same reasoning as the
        # unsafe check above - no point spending a cache lookup, let alone
        # retrieval/generation, on a query the corpus was never going to be
        # able to ground an answer in.
        is_off_topic, off_topic_similarity = guardrails.check_off_topic(query_embedding)
        if is_off_topic:
            total_latency = (time.time() - pipeline_start) * 1000
            latency_breakdown["total"] = total_latency
            return QueryResponse(
                query=request.query_text,
                answer=build_off_topic_response(request.language),
                retrieved_documents=[],
                confidence=0.0,
                latency_breakdown={k: f"{v:.1f}ms" for k, v in latency_breakdown.items()},
                status="declined",
                declined=True,
                decline_reason="off_topic",
            )

        # ====== Step 2b: Semantic Cache Lookup ======
        # Checked BEFORE retrieval, not after like the literal cache - a hit
        # here skips retrieval AND generation, not just generation. Fuzzy
        # match on query embedding similarity (calibrated threshold, see
        # scripts/calibrate_semantic_cache_threshold.py) catches different
        # phrasing of the same intent that the literal cache's exact-match
        # key would never hit.
        start = time.time()
        semantic_payload, semantic_similarity = semantic_cache.lookup(query_embedding)
        latency_breakdown["semantic_cache_lookup"] = (time.time() - start) * 1000

        if semantic_payload is not None:
            total_latency = (time.time() - pipeline_start) * 1000
            latency_breakdown["total"] = total_latency
            return QueryResponse(
                query=request.query_text,
                answer=semantic_payload["answer"],
                retrieved_documents=semantic_payload["retrieved_documents"],
                confidence=float(semantic_payload["grounding_score"]),
                latency_breakdown={k: f"{v:.1f}ms" for k, v in latency_breakdown.items()},
                status="ok",
                cache_hit=True,
                backend="cache",
                cache_type="semantic",
                semantic_similarity=semantic_similarity,
                hedged=False,  # hedges are never cached, see Step 8 below
            )

        # ====== Step 3: Parallel Retrieval ======
        # Dense and sparse run concurrently via to_thread - wall time should
        # track max(dense, sparse), not their sum. Both branches are timed
        # individually too so a regression in either is visible without
        # having to re-derive it from the wall clock.
        async def _timed(label, fn, *args, **kwargs):
            t0 = time.time()
            result = await asyncio.to_thread(fn, *args, **kwargs)
            latency_breakdown[label] = (time.time() - t0) * 1000
            return result

        start = time.time()
        dense_results, bm25_results = await asyncio.gather(
            _timed("retrieval_dense", chroma_service.query, query_embedding.tolist(), top_k=10),
            _timed("retrieval_sparse", bm25_service.query, request.query_text, top_k=10),
        )
        latency_breakdown["retrieval"] = (time.time() - start) * 1000

        # ====== Step 4: Merge Results ======
        start = time.time()
        retrieved_docs = merge_and_rank(
            dense_results,
            bm25_results,
            top_k=request.top_k,
            target_language=request.language,
        )
        latency_breakdown["merge"] = (time.time() - start) * 1000

        formatted_docs = [
            {
                "doc_id": doc["doc_id"],
                "content": doc["content"][:200],  # Truncate for response
                "score": float(doc["final_score"]),
                "chunking_strategy": doc.get("chunking_strategy"),  # None = whole passage, see src/chunking/
            }
            for doc in retrieved_docs
        ]

        # ====== Step 4c: Literal Answer Cache Lookup ======
        # Keyed on (query, retrieved doc set) - retrieval above always runs
        # fresh, so a corpus change that alters which docs a query retrieves
        # changes the key automatically instead of serving stale context.
        cache_key = make_cache_key(request.query_text, [doc["doc_id"] for doc in retrieved_docs])
        cache_hit = False
        cache_type = None
        is_hedged = False  # never True on a cache hit - hedges are never cached, see below
        start = time.time()
        cached = answer_cache.get(cache_key)
        latency_breakdown["cache_lookup"] = (time.time() - start) * 1000

        if cached is not None:
            cache_hit = True
            cache_type = "literal"
            answer = cached["answer"]
            grounding_score = cached["grounding_score"]
            backend = "cache"
        else:
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
                use_fast_path=(elapsed_ms < 100),  # If we have time budget, use local LLM
                language=request.language,
            )
            backend = generation_service.last_backend

            latency_breakdown["generation"] = (time.time() - start) * 1000

            # ====== Step 7: Grounding Check ======
            start = time.time()
            grounding_score = guardrails.check_grounding(
                answer=answer,
                retrieved_docs=context_docs
            )
            latency_breakdown["grounding"] = (time.time() - start) * 1000

            # ====== Step 8: Validate Answer ======
            is_valid = guardrails.validate_answer(answer)
            if not is_valid:
                # Model explicitly refused / gave an empty or refusal-phrase
                # non-answer.
                answer = "I could not find a clear answer to your question. Please try rephrasing."
                grounding_score = 0.5
            elif grounding_score < ANSWER_CACHE_MIN_GROUNDING:
                # Passed validate_answer but isn't actually grounded in the
                # retrieved context - the confident-hallucination case, not
                # the "model said it doesn't know" case.
                answer = build_low_grounding_response(request.language)
                is_hedged = True

            # Never cache a hedge or refusal response as if it were the answer.
            if is_valid and not is_hedged and grounding_score >= ANSWER_CACHE_MIN_GROUNDING:
                answer_cache.set(cache_key, {"answer": answer, "grounding_score": grounding_score})
                semantic_cache.set(cache_key, query_embedding, {
                    "answer": answer,
                    "grounding_score": grounding_score,
                    "retrieved_documents": formatted_docs,
                })

        # ====== Step 9: Format Response ======
        total_latency = (time.time() - pipeline_start) * 1000
        latency_breakdown["total"] = total_latency

        return QueryResponse(
            query=request.query_text,
            answer=answer,
            retrieved_documents=formatted_docs,
            confidence=float(grounding_score),
            latency_breakdown={k: f"{v:.1f}ms" for k, v in latency_breakdown.items()},
            status="ok",
            cache_hit=cache_hit,
            backend=backend,
            cache_type=cache_type,
            hedged=is_hedged,
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if DEBUG:
            print(f"✗ Query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query_endpoint(request: QueryRequest):
    return await _handle_query(request)


@app.post("/demo/query", response_model=QueryResponse, dependencies=[Depends(verify_demo_rate_limit)])
async def demo_query_endpoint(request: QueryRequest):
    """Same pipeline as /query - no RAG_API_KEY required, gated by
    verify_demo_rate_limit instead. Exists so the public dashboard works
    for a judge with zero setup, without re-exposing RAG_API_KEY."""
    return await _handle_query(request)


async def _maybe_send(websocket, payload):
    """websocket may be None for a speculative (headless) pipeline run - see
    _run_query_pipeline's docstring."""
    if websocket is not None:
        await websocket.send_json(payload)


async def _run_query_pipeline(websocket, query_text, top_k, target_language, pipeline_start, latency_breakdown):
    """Shared RAG pipeline: embedding -> semantic cache -> retrieval -> merge
    -> literal cache / generation -> grounding -> final. Streams one JSON
    event per stage over `websocket`. Used by the text-query and audio-query
    (post-STT) branches of ws_query_endpoint's loop, so a mic recording gets
    the exact same live per-stage progress and latency breakdown as a typed
    query - `latency_breakdown` may already carry an upstream "stt" entry
    from the caller, folded into the final "total".

    `websocket` may also be None: speculative_query fires this same function
    headlessly (no stage events sent anywhere) on live-caption prefixes
    while the user is still talking, purely to populate answer_cache /
    semantic_cache ahead of time. No separate reconciliation path exists -
    when the real post-stop audio_query runs this function again with the
    actual Sarvam transcript, the semantic_cache lookup a few lines down is
    the reconciliation: if the speculative guess's embedding was close
    enough (cosine >= SEMANTIC_CACHE_SIMILARITY_THRESHOLD) to the real
    transcript's, this becomes an instant cache hit."""
    if check_unsafe(query_text):
        latency_breakdown["total"] = (time.time() - pipeline_start) * 1000
        await _maybe_send(websocket, {
            "stage": "final",
            "query": query_text,
            "answer": build_unsafe_response(target_language),
            "confidence": 0.0,
            "cache_hit": False,
            "hedged": False,
            "declined": True,
            "decline_reason": "unsafe",
            "backend": None,
            "retrieved_documents": [],
            "latency_breakdown": {k: round(v, 1) for k, v in latency_breakdown.items()},
        })
        return

    await _maybe_send(websocket, {"stage": "embedding", "status": "start"})
    start = time.time()
    query_embedding = await asyncio.to_thread(embedding_service.embed_query, query_text)
    latency_breakdown["embedding"] = (time.time() - start) * 1000
    await _maybe_send(websocket, {
        "stage": "embedding", "status": "done",
        "latency_ms": round(latency_breakdown["embedding"], 1),
    })

    is_off_topic, off_topic_similarity = guardrails.check_off_topic(query_embedding)
    if is_off_topic:
        latency_breakdown["total"] = (time.time() - pipeline_start) * 1000
        await _maybe_send(websocket, {
            "stage": "final",
            "query": query_text,
            "answer": build_off_topic_response(target_language),
            "confidence": 0.0,
            "cache_hit": False,
            "hedged": False,
            "declined": True,
            "decline_reason": "off_topic",
            "backend": None,
            "retrieved_documents": [],
            "latency_breakdown": {k: round(v, 1) for k, v in latency_breakdown.items()},
        })
        return

    start = time.time()
    semantic_payload, semantic_similarity = await asyncio.to_thread(semantic_cache.lookup, query_embedding)
    latency_breakdown["semantic_cache_lookup"] = (time.time() - start) * 1000

    if semantic_payload is not None:
        latency_breakdown["total"] = (time.time() - pipeline_start) * 1000
        await _maybe_send(websocket, {
            "stage": "final",
            "query": query_text,
            "answer": semantic_payload["answer"],
            "confidence": float(semantic_payload["grounding_score"]),
            "cache_hit": True,
            "cache_type": "semantic",
            "hedged": False,  # hedges are never cached, see _run_query_pipeline's gating below
            "semantic_similarity": semantic_similarity,
            "backend": "cache",
            "retrieved_documents": semantic_payload["retrieved_documents"],
            "latency_breakdown": {k: round(v, 1) for k, v in latency_breakdown.items()},
        })
        return

    await _maybe_send(websocket, {"stage": "retrieval", "status": "start"})
    async def _timed(label, fn, *args, **kwargs):
        t0 = time.time()
        result = await asyncio.to_thread(fn, *args, **kwargs)
        latency_breakdown[label] = (time.time() - t0) * 1000
        return result

    start = time.time()
    dense_results, bm25_results = await asyncio.gather(
        _timed("retrieval_dense", chroma_service.query, query_embedding.tolist(), top_k=10),
        _timed("retrieval_sparse", bm25_service.query, query_text, top_k=10),
    )
    latency_breakdown["retrieval"] = (time.time() - start) * 1000
    await _maybe_send(websocket, {
        "stage": "retrieval", "status": "done",
        "latency_ms": round(latency_breakdown["retrieval"], 1),
        "dense_ms": round(latency_breakdown["retrieval_dense"], 1),
        "sparse_ms": round(latency_breakdown["retrieval_sparse"], 1),
        "dense_count": len(dense_results), "bm25_count": len(bm25_results),
    })

    start = time.time()
    retrieved_docs = merge_and_rank(dense_results, bm25_results, top_k=top_k, target_language=target_language)
    latency_breakdown["merge"] = (time.time() - start) * 1000
    formatted_docs = [
        {
            "doc_id": doc["doc_id"],
            "content": doc["content"][:200],
            "score": float(doc["final_score"]),
            "chunking_strategy": doc.get("chunking_strategy"),  # None = whole passage, see src/chunking/
        }
        for doc in retrieved_docs
    ]
    await _maybe_send(websocket, {
        "stage": "merge", "status": "done",
        "latency_ms": round(latency_breakdown["merge"], 1),
        "documents": formatted_docs,
    })

    context_docs = [doc["content"] for doc in retrieved_docs]

    cache_key = make_cache_key(query_text, [doc["doc_id"] for doc in retrieved_docs])
    cache_hit = False
    cache_type = None
    is_hedged = False  # never True on a cache hit - hedges are never cached, see below
    start = time.time()
    cached = answer_cache.get(cache_key)
    latency_breakdown["cache_lookup"] = (time.time() - start) * 1000

    if cached is not None:
        cache_hit = True
        cache_type = "literal"
        answer = cached["answer"]
        grounding_score = cached["grounding_score"]
        backend = "cache"
        await _maybe_send(websocket, {
            "stage": "generation", "status": "done",
            "latency_ms": 0.0, "answer": answer, "cache_hit": True, "backend": backend,
        })
        await _maybe_send(websocket, {
            "stage": "grounding", "status": "done",
            "latency_ms": 0.0, "score": grounding_score,
        })
    else:
        await _maybe_send(websocket, {"stage": "generation", "status": "start"})
        start = time.time()
        answer_parts = []
        backend = None
        first_token_elapsed_ms = None
        async for event in generation_service.stream_generate(query_text, context_docs, target_language):
            if first_token_elapsed_ms is None:
                first_token_elapsed_ms = (time.time() - start) * 1000
            answer_parts.append(event["delta"])
            backend = event["backend"]
            await _maybe_send(websocket, {
                "stage": "generation", "status": "token",
                "delta": event["delta"], "backend": backend,
            })
        # Pure string cleanup backstop (never a second LLM call - see
        # generation_service.clean_answer_boilerplate) - applied to the
        # aggregate used for grounding/caching/the final event. The
        # already-streamed per-token preview may briefly show the
        # unstripped opener; that's a cosmetic tradeoff, not a correctness
        # one, per Markdown/natural_output_and_accuracy_prompt.md.
        answer = clean_answer_boilerplate("".join(answer_parts).strip())
        latency_breakdown["generation"] = (time.time() - start) * 1000
        if first_token_elapsed_ms is not None:
            latency_breakdown["first_token"] = first_token_elapsed_ms
        await _maybe_send(websocket, {
            "stage": "generation", "status": "done",
            "latency_ms": round(latency_breakdown["generation"], 1),
            "first_token_ms": round(first_token_elapsed_ms or 0, 1),
            "answer": answer, "backend": backend,
        })

        await _maybe_send(websocket, {"stage": "grounding", "status": "start"})
        start = time.time()
        grounding_score = await asyncio.to_thread(guardrails.check_grounding, answer, context_docs)
        latency_breakdown["grounding"] = (time.time() - start) * 1000
        await _maybe_send(websocket, {
            "stage": "grounding", "status": "done",
            "latency_ms": round(latency_breakdown["grounding"], 1),
            "score": grounding_score,
        })

        is_valid = guardrails.validate_answer(answer)
        if not is_valid:
            # Model explicitly refused / gave an empty or refusal-phrase
            # non-answer.
            answer = "I could not find a clear answer to your question. Please try rephrasing."
            grounding_score = 0.5
        elif grounding_score < ANSWER_CACHE_MIN_GROUNDING:
            # Passed validate_answer (not empty/refusal-phrase) but isn't
            # actually grounded in the retrieved context - the confident-
            # hallucination case, not the "model said it doesn't know" case.
            # Reproduces the audit's 0.0868-grounding Gujarati finding.
            answer = build_low_grounding_response(target_language)
            is_hedged = True

        # Never cache a hedge or refusal response as if it were the answer.
        if is_valid and not is_hedged and grounding_score >= ANSWER_CACHE_MIN_GROUNDING:
            answer_cache.set(cache_key, {"answer": answer, "grounding_score": grounding_score})
            semantic_cache.set(cache_key, query_embedding, {
                "answer": answer,
                "grounding_score": grounding_score,
                "retrieved_documents": formatted_docs,
            })

    latency_breakdown["total"] = (time.time() - pipeline_start) * 1000
    await _maybe_send(websocket, {
        "stage": "final",
        "query": query_text,
        "answer": answer,
        "confidence": float(grounding_score),
        "cache_hit": cache_hit,
        "cache_type": cache_type,
        "hedged": is_hedged,
        "backend": backend,
        "retrieved_documents": formatted_docs,
        "latency_breakdown": {k: round(v, 1) for k, v in latency_breakdown.items()},
    })


async def _run_speculative_pipeline(query_text, target_language):
    """Headless background run of _run_query_pipeline on a not-yet-final
    transcript prefix - see that function's docstring for how this ties
    back into the real post-stop query via the semantic cache. Errors are
    swallowed (best-effort, nothing downstream is waiting on this) except
    CancelledError, which must propagate so asyncio.Task.cancel() actually
    works when a newer prefix supersedes this one."""
    try:
        if DEBUG:
            print(f"[speculative] running: {query_text[:60]!r}")
        await _run_query_pipeline(None, query_text, TOP_K_FINAL, target_language, time.time(), {})
        if DEBUG:
            print(f"[speculative] done: {query_text[:60]!r}")
    except asyncio.CancelledError:
        if DEBUG:
            print(f"[speculative] cancelled: {query_text[:60]!r}")
        raise
    except Exception as e:
        if DEBUG:
            print(f"[speculative] failed: {query_text[:60]!r} - {e}")


async def _relay_realtime_stt(websocket, session, target_language, state):
    """Background task that runs for the duration of one recording: reads
    events off the live Sarvam realtime STT session and relays
    transcript.partial/transcript.final events back to the dashboard (this
    is what drives the live "what's being heard" preview now - straight
    from the same engine that produces the answer, not a separate guess
    from the browser's own speech recognizer). Accumulates final text per
    utterance into `state["final_parts"]`, and fires a speculative pipeline
    run (see _run_speculative_pipeline) once a partial stops changing
    between two consecutive events, so retrieval+generation for the likely
    final question is already underway before the user even stops talking.
    Sets state["ended"] when Sarvam closes the session (recording finished)
    or reports a fatal error."""
    last_partial = None
    speculative_task = None
    try:
        async for msg in session:
            event = getattr(msg, "event", None)
            if event == "transcript.partial":
                await websocket.send_json({"stage": "stt", "status": "partial", "transcript": msg.text})
                if msg.text == last_partial and len(msg.text.split()) >= 3:
                    if speculative_task is None or speculative_task.done():
                        speculative_task = asyncio.create_task(
                            _run_speculative_pipeline(msg.text, target_language)
                        )
                last_partial = msg.text
            elif event == "transcript.final":
                state["final_parts"].append(msg.text)
                if getattr(msg, "language", None):
                    state["detected_language"] = msg.language
            elif event == "error":
                state["error"] = getattr(msg, "message", "unknown error")
                if getattr(msg, "is_fatal", False):
                    state["ended"].set()
            elif event == "session.end":
                state["ended"].set()
                break
    except asyncio.CancelledError:
        raise
    except Exception as e:
        if DEBUG:
            print(f"[realtime_stt] listener error: {e}")
        state["error"] = str(e)
        state["ended"].set()
    finally:
        if speculative_task is not None and not speculative_task.done():
            speculative_task.cancel()


async def _ws_handler(websocket: WebSocket, require_auth: bool = True):
    """
    Streaming version of /query for the dashboard: pushes one JSON event
    per pipeline stage (start/done + timing) instead of waiting for the full
    response, so the UI can show live progress.

    Browsers' native WebSocket API can't set custom headers, so auth happens
    after the handshake: the client's first message must be
    {"type": "auth", "api_key": ...}. A query-string api_key was considered
    and rejected - it would land in uvicorn's access log and browser history
    in cleartext.

    Shared by both /ws/query (require_auth=True) and /ws/demo
    (require_auth=False, rate-limited instead - see verify_demo_rate_limit
    and the _check_demo_rate_limit closure below). Identical pipeline
    either way, only the access gate differs.
    """
    await websocket.accept()

    if require_auth and RAG_API_KEY:
        try:
            auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        except Exception:
            await websocket.close(code=4401, reason="Auth message required")
            return
        submitted_key = auth_msg.get("api_key") if isinstance(auth_msg, dict) else None
        if not secrets.compare_digest(submitted_key or "", RAG_API_KEY):
            await websocket.close(code=4401, reason="Invalid API key")
            return

    # Rate limit is per query submitted, not per raw WS message - a voice
    # recording sends many small audio_chunk messages for one logical query,
    # and those shouldn't count against the same budget as a judge typing
    # several separate questions.
    async def _check_demo_rate_limit() -> bool:
        if require_auth:
            return True
        if demo_rate_limiter.allow(_client_ip(websocket)):
            return True
        await websocket.send_json({"stage": "error", "message": DEMO_RATE_LIMIT_MESSAGE})
        return False

    await websocket.send_json({"stage": "auth", "status": "ok"})

    # At most one speculative (pre-generation, off transcript prefixes)
    # pipeline run in flight per connection - a newer prefix or the real
    # audio_query/audio_stream_end supersedes whatever's still running,
    # rather than piling up concurrent GPU/Groq work for guesses that are
    # already stale.
    speculative_task = None

    # Live realtime STT session state (audio_stream_start/audio_chunk/
    # audio_stream_end) - one recording at a time per connection. See
    # RealtimeSttSession and _relay_realtime_stt.
    stt_session = None
    stt_listen_task = None
    stt_stream_state = None
    # Language the in-progress recording was opened with (from
    # audio_stream_start's `language`) - audio_stream_end carries no
    # `language` field of its own (see dashboard's ws.send for that message),
    # so re-reading target_language off *that* message at the top of the loop
    # would silently fall back to "en" regardless of what language was
    # actually recorded in. Must be read back at audio_stream_end time
    # instead of relying on the loop-local target_language variable.
    stt_target_language = None

    async def _close_stt_session():
        nonlocal stt_session, stt_listen_task, stt_stream_state, stt_target_language
        if stt_listen_task is not None and not stt_listen_task.done():
            stt_listen_task.cancel()
        if stt_session is not None:
            try:
                await stt_session.__aexit__(None, None, None)
            except Exception:
                pass
        stt_session = None
        stt_listen_task = None
        stt_stream_state = None
        stt_target_language = None

    try:
        while True:
            request = await websocket.receive_json()
            msg_type = request.get("type") or "query"
            top_k = request.get("top_k") or TOP_K_FINAL
            target_language = request.get("language") or "en"

            if msg_type == "audio_stream_start":
                if not await _check_demo_rate_limit():
                    continue
                await _close_stt_session()
                if speculative_task is not None and not speculative_task.done():
                    speculative_task.cancel()
                    speculative_task = None
                try:
                    # Explicit language, from the dashboard's language
                    # selector - Sarvam's realtime "auto" detection was
                    # tested and found unreliable on short single-sentence
                    # clips (misdetected real Hindi speech as English and
                    # transliterated it - see conversation). An explicit
                    # code transcribes correctly in the language's own
                    # script; RealtimeSttSession falls back to "auto" only
                    # for a code this system doesn't have a Sarvam mapping
                    # for.
                    stt_session = await RealtimeSttSession(target_language).__aenter__()
                except RuntimeError as e:
                    await websocket.send_json({"stage": "error", "message": str(e)})
                    continue
                except Exception as e:
                    if DEBUG:
                        print(f"[ws_query] Failed to open realtime STT session: {e}")
                    await websocket.send_json({"stage": "error", "message": "Speech-to-text provider connection failed"})
                    continue
                stt_target_language = target_language
                stt_stream_state = {"final_parts": [], "ended": asyncio.Event(), "error": None, "detected_language": None}
                stt_listen_task = asyncio.create_task(
                    _relay_realtime_stt(websocket, stt_session, target_language, stt_stream_state)
                )
                await websocket.send_json({"stage": "stt", "status": "start"})
                continue

            if msg_type == "audio_chunk":
                if stt_session is None:
                    continue
                chunk_b64 = request.get("chunk")
                if chunk_b64:
                    try:
                        await stt_session.send_chunk(chunk_b64)
                    except Exception as e:
                        if DEBUG:
                            print(f"[ws_query] audio_chunk send failed: {e}")
                continue

            if msg_type == "audio_stream_end":
                if stt_session is None or stt_stream_state is None:
                    await websocket.send_json({"stage": "error", "message": "No active recording session"})
                    continue

                pipeline_start = time.time()
                latency_breakdown = {}
                stt_start = time.time()
                try:
                    await stt_session.end()
                    try:
                        await asyncio.wait_for(stt_stream_state["ended"].wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        if DEBUG:
                            print("[ws_query] Timed out waiting for realtime STT session end")
                except Exception as e:
                    if DEBUG:
                        print(f"[ws_query] Error ending realtime STT session: {e}")

                query_text = " ".join(p for p in stt_stream_state["final_parts"] if p).strip()
                stream_error = stt_stream_state["error"]
                detected_sarvam_code = stt_stream_state["detected_language"]
                # audio_stream_end carries no `language` field (see
                # dashboard's ws.send for that message type), so the
                # target_language read at the top of the loop for *this*
                # message defaulted to "en" - use the language the recording
                # was actually opened with instead. Must read this back
                # before _close_stt_session() clears it.
                target_language = stt_target_language or target_language
                await _close_stt_session()

                # Sarvam auto-detected the actual spoken language (see the
                # "auto" comment above) - use it for retrieval's
                # language-match ranking and the LLM's answer-language
                # instruction, instead of the unreliable client-sent
                # target_language default. Falls back to target_language
                # only if detection didn't report anything.
                if detected_sarvam_code:
                    target_language = sarvam_code_to_language(detected_sarvam_code, default=target_language)

                latency_breakdown["stt"] = (time.time() - stt_start) * 1000

                if stream_error:
                    await websocket.send_json({"stage": "error", "message": f"Speech-to-text error: {stream_error}"})
                    continue
                if not query_text:
                    await websocket.send_json({
                        "stage": "error",
                        "message": "Could not transcribe audio - empty transcript",
                    })
                    continue

                await websocket.send_json({
                    "stage": "stt", "status": "done",
                    "latency_ms": round(latency_breakdown["stt"], 1),
                    "transcript": query_text,
                    "language_code": detected_sarvam_code,
                })

                if speculative_task is not None and not speculative_task.done():
                    speculative_task.cancel()
                    speculative_task = None

                # No rate-limit check here - this recording already spent its
                # budget at audio_stream_start (see there for why: the STT
                # cost happens regardless of whether the pipeline ever runs,
                # so gating only here would let someone rack up real Sarvam
                # cost per recording without it ever counting against them).
                try:
                    await _run_query_pipeline(
                        websocket, query_text, top_k, target_language, pipeline_start, latency_breakdown
                    )
                except Exception as e:
                    if DEBUG:
                        print(f"[ws_query] Pipeline error: {e}")
                    await websocket.send_json({"stage": "error", "message": "Internal server error"})
                continue

            if msg_type == "speculative_query":
                if speculative_task is not None and not speculative_task.done():
                    speculative_task.cancel()
                text = (request.get("text") or "").strip()
                if text:
                    speculative_task = asyncio.create_task(_run_speculative_pipeline(text, target_language))
                continue

            if msg_type == "audio_query":
                if speculative_task is not None and not speculative_task.done():
                    speculative_task.cancel()
                    speculative_task = None

                audio_b64 = request.get("audio_b64")
                if not audio_b64:
                    await websocket.send_json({"stage": "error", "message": "audio_b64 is required"})
                    continue

                # Checked before STT runs, not after - the Sarvam STT call
                # below costs real money regardless of whether a query ever
                # completes, so gating only around _run_query_pipeline would
                # let that cost be run up for free (same reasoning as the
                # audio_stream_start check above).
                if not await _check_demo_rate_limit():
                    continue

                pipeline_start = time.time()
                latency_breakdown = {}
                try:
                    await websocket.send_json({"stage": "stt", "status": "start"})
                    audio_bytes = base64.b64decode(audio_b64)
                    content_type = request.get("content_type") or "audio/webm"
                    start = time.time()
                    stt_result = await asyncio.to_thread(
                        stt_service.transcribe, audio_bytes, "recording.webm", target_language, content_type
                    )
                    latency_breakdown["stt"] = (time.time() - start) * 1000

                    query_text = stt_result["transcript"]
                    if not query_text:
                        await websocket.send_json({
                            "stage": "error",
                            "message": "Could not transcribe audio - empty transcript",
                        })
                        continue

                    await websocket.send_json({
                        "stage": "stt", "status": "done",
                        "latency_ms": round(latency_breakdown["stt"], 1),
                        "transcript": query_text,
                        "language_code": stt_result["language_code"],
                    })

                    await _run_query_pipeline(
                        websocket, query_text, top_k, target_language, pipeline_start, latency_breakdown
                    )
                except RuntimeError as e:
                    await websocket.send_json({"stage": "error", "message": str(e)})
                except httpx.HTTPStatusError as e:
                    if DEBUG:
                        print(f"[ws_query] Sarvam STT request failed: {e}")
                    await websocket.send_json({"stage": "error", "message": "Speech-to-text provider request failed"})
                except Exception as e:
                    if DEBUG:
                        print(f"[ws_query] Audio pipeline error: {e}")
                    await websocket.send_json({"stage": "error", "message": "Internal server error"})
                continue

            if speculative_task is not None and not speculative_task.done():
                speculative_task.cancel()
                speculative_task = None

            query_text = (request.get("query_text") or "").strip()
            if not query_text:
                await websocket.send_json({"stage": "error", "message": "Query cannot be empty"})
                continue
            if len(query_text) > MAX_QUERY_TEXT_LENGTH:
                await websocket.send_json({
                    "stage": "error",
                    "message": f"Query is too long (max {MAX_QUERY_TEXT_LENGTH} characters)",
                })
                continue

            if not await _check_demo_rate_limit():
                continue

            pipeline_start = time.time()
            latency_breakdown = {}

            try:
                await _run_query_pipeline(
                    websocket, query_text, top_k, target_language, pipeline_start, latency_breakdown
                )
            except Exception as e:
                if DEBUG:
                    print(f"[ws_query] Pipeline error: {e}")
                await websocket.send_json({"stage": "error", "message": "Internal server error"})
    except WebSocketDisconnect:
        pass
    finally:
        if speculative_task is not None and not speculative_task.done():
            speculative_task.cancel()
        await _close_stt_session()


@app.websocket("/ws/query")
async def ws_query_endpoint(websocket: WebSocket):
    await _ws_handler(websocket, require_auth=True)


@app.websocket("/ws/demo")
async def ws_demo_endpoint(websocket: WebSocket):
    """Same pipeline as /ws/query (text and voice both work) - no
    RAG_API_KEY required, gated by a per-IP rate limit instead. This is
    what the public dashboard connects to by default."""
    await _ws_handler(websocket, require_auth=False)


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
    try:
        # Read audio file
        audio_bytes = await audio.read()

        # ====== STT (Sarvam AI, Saarika model) ======
        stt_start = time.time()
        try:
            stt_result = await asyncio.to_thread(
                stt_service.transcribe, audio_bytes, audio.filename, language, audio.content_type
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except httpx.HTTPStatusError as e:
            if DEBUG:
                print(f"✗ Sarvam STT request failed: {e}")
            raise HTTPException(status_code=502, detail="Speech-to-text provider request failed")
        stt_latency_ms = (time.time() - stt_start) * 1000

        query_text = stt_result["transcript"]
        if not query_text:
            raise HTTPException(status_code=422, detail="Could not transcribe audio - empty transcript")

        # ====== Reuse query endpoint ======
        request = QueryRequest(query_text=query_text, language=language)
        response = await query_endpoint(request)
        response.latency_breakdown["stt"] = f"{stt_latency_ms:.1f}ms"
        return response

    except HTTPException:
        raise
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
    import sys
    import uvicorn

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║   HH Goa 2026: Voice-Enabled RAG System                    ║
    ║   Server starting...                                        ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main_app:app",
        # 127.0.0.1 for local dev. /dashboard no longer server-injects
        # RAG_API_KEY (see dashboard() above) so this is safe to change to
        # 0.0.0.0 for deployment - the key is only ever typed client-side
        # and sent over the WS auth frame, never rendered into HTML.
        host="127.0.0.1",
        port=8000,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )

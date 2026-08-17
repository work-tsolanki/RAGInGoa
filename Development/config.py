import os
from dotenv import load_dotenv

load_dotenv()

# Defaults are relative paths for local dev. In production (Fly), these point
# at a mounted volume (see fly.toml's [mounts] + [env]) - the ~4.6GB corpus
# lives on the volume, not baked into the image, since a 9GB image proved too
# large to reliably pull onto a Fly host within its provisioning timeout.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
BM25S_INDEX_DIR = os.getenv("BM25S_INDEX_DIR", "bm25s_index_full")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "mock")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "mock")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "mock")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
RAG_API_KEY = os.getenv("RAG_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384

TOP_K_RETRIEVAL = 10
TOP_K_FINAL = 5

MAX_LATENCY_MS = 200
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

USE_LOCAL_LLM = True
USE_CLAUDE_FALLBACK = True

LOCAL_LLM_MODEL_PATH = os.getenv(
    "LOCAL_LLM_MODEL_PATH",
    "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)
LOCAL_LLM_N_CTX = 4096
LOCAL_LLM_MAX_TOKENS = 200
# -1 = offload all layers to GPU. Ignored (falls back to CPU) if llama-cpp-python
# wasn't built with CUDA support.
LOCAL_LLM_N_GPU_LAYERS = int(os.getenv("LOCAL_LLM_N_GPU_LAYERS", "-1"))

# Groq: hosted llama-3.1-8b-instant, same weights as the local model but
# ~840 tok/s vs local's ~130 tok/s. Primary generation backend - see
# GENERATION_BACKEND_ORDER. Falls through to local/Claude if unset or if a
# request fails before any tokens are streamed back.
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MAX_TOKENS = 200  # matches LOCAL_LLM_MAX_TOKENS - same model, same answer-length expectations
GROQ_TEMPERATURE = 0.1  # lowered from 0.3 (Phase 3 hardening): the live deploy's own
                         # benchmark caught the same query ("How to apply for a passport")
                         # scoring grounded (0.986) locally and hedged (0.352) on a repeat
                         # run - phrasing variance from temperature feeding into the
                         # grounding-score jitter, not a retrieval difference. Lower temp
                         # trades a little answer-phrasing variety for hedge consistency,
                         # which matters more when the same question can be asked twice
                         # live by a judge.

# Cerebras: originally planned as a same-weights-different-silicon speed
# test vs Groq (both running Llama-3.1-8B). That model isn't available on
# this account/tier - confirmed via client.models.list(), which returned
# only gemma-4-31b, zai-glm-4.7, gpt-oss-120b. Using gpt-oss-120b instead,
# which makes this a full accuracy+speed comparison (different, larger
# model), not the narrow silicon-only test the plan called for.
CEREBRAS_MODEL = "gpt-oss-120b"
CEREBRAS_MAX_TOKENS = 200
CEREBRAS_TEMPERATURE = 0.3

# Ordered generation backend priority. Each name must have a corresponding
# generate_<name>-style method on GenerationService. Configurable here
# without touching the fallback-chain code itself. Groq primary, local
# llama.cpp and Claude as fallbacks if Groq is unset or fails before
# streaming any tokens back.
GENERATION_BACKEND_ORDER = ["groq", "local", "claude"]
_backend_order_override = os.getenv("GENERATION_BACKEND_ORDER_OVERRIDE")
if _backend_order_override:
    GENERATION_BACKEND_ORDER = [b.strip() for b in _backend_order_override.split(",") if b.strip()]

# Per-backend request timeout + SDK retry cap. None of the hosted-API
# clients (groq, cerebras, anthropic) had a timeout set before this - each
# just used its SDK's own default (tens of seconds), so a single degraded
# Groq request could block the whole fallback chain from ever reaching
# local/Claude. Measured live (benchmark/percentile_batch_results.json):
# generation P50=261ms, P70=309ms, but P100=13851ms - a ~45x blowout from
# exactly this gap, not from the model itself being slow. These timeouts
# are set with wide margin above the real P70 (10x+) so a normal request is
# never at risk of a false-fallback, while still bounding the worst case
# to single-digit seconds instead of double digits. max_retries=0 because
# GENERATION_BACKEND_ORDER is already our retry policy (try the next
# backend) - stacking the SDK's own internal retry (2, by default) on top
# would multiply the timeout instead of shortening the tail.
GROQ_REQUEST_TIMEOUT_S = 3.0
CEREBRAS_REQUEST_TIMEOUT_S = 3.0
CLAUDE_REQUEST_TIMEOUT_S = 5.0  # last-resort fallback before extractive; a little
                                 # more headroom since correctness matters more here
GENERATION_SDK_MAX_RETRIES = 0

# Sarvam AI: Speech-to-Text (Saarika model) for /query_audio. REST API, no
# official Python SDK - see src/stt_service.py. Get a key at
# https://dashboard.sarvam.ai (free tier available). Left as "mock" (the
# .env.example/.env default), /query_audio fails fast with a clear error
# instead of silently returning a fake transcript.
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saarika:v2.5")
SARVAM_BASE_URL = "https://api.sarvam.ai"

# Bumped whenever generation_service._build_prompt's instruction/few-shot
# content changes meaningfully (not for unrelated edits). Folded into both
# answer_cache's key (src/answer_cache.py) and semantic_cache's stored
# entries (src/semantic_cache.py) so a prompt-style change can't serve an
# old-style cached answer next to fresh new-style ones - old entries become
# unreachable immediately (new key/version never matches), no manual flush
# needed. See Markdown/natural_output_and_accuracy_prompt.md.
PROMPT_VERSION = "v4-extended-tokens-gu-ta-bn"

# Answer cache: skips generation + grounding on a repeat (query, retrieved
# doc set). Only caches answers that passed guardrails.validate_answer().
ANSWER_CACHE_MAX_SIZE = 1000
ANSWER_CACHE_TTL_SECONDS = 86400  # 24h - answers about deadlines/rules can go stale
# Calibrated against 38 hand-labeled (query, answer, grounding_score) triples,
# labeled blind to the score - see benchmark/grounding_calibration_raw.json
# and scripts/analyze_grounding_calibration.py. TRUE (grounded/correct) and
# FALSE (ungrounded/wrong) do NOT separate cleanly: 4/20 FALSE answers score
# >=0.92 (confidently wrong due to topic/homonym retrieval mismatches, e.g.
# "gst registration fees" answered from a vehicle-registration passage at
# 0.9989) and 1/18 TRUE answer (an honest partial/hedged response) scores as
# low as 0.1584. This is real information about the cross-encoder's limited
# discriminative power on this domain, not a calibration failure - no
# threshold catches those 4 false positives without also rejecting far more
# true positives (pushing to 0.999 only saves 1 more false positive at the
# cost of a third of all good answers). 0.7 is the point past which raising
# the threshold stops helping: retains 94% of true positives, correctly
# excludes 80% of false ones (mostly soft refusals like "the context does
# not provide..." that slip past guardrails.validate_answer()'s exact-phrase
# check). The remaining ~4 confidently-wrong-answer risk is accepted and
# documented, not silently ignored - would need a retrieval-relevance gate
# or a better grounding check to close, not a threshold tweak.
#
# UPDATE (grounding-gate rollout, see Markdown/grounding_gate_and_chunking_
# implementation.md): as of this change, this threshold also gates the
# *returned* answer, not just caching (main_app.py - below it, both /query
# and the WS pipeline replace a sub-threshold answer with
# guardrails.build_low_grounding_response() instead of returning it as-is).
# Live-testing that rollout surfaced a real, since-accepted tradeoff: this
# calibration was run against the OLD, more literal/extractive prompt.
# generation_service.py's later natural-output rewrite (deliberately) makes
# answers more paraphrased, which lowers cross-encoder grounding scores for
# some genuinely-correct answers purely from reduced lexical overlap with
# the source (e.g. a real "how to apply for a passport" answer dropped from
# ~0.999 under the old literal-copy style to ~0.35 paraphrased, and now
# hedges under this threshold). Decision: leave the threshold as-is rather
# than re-calibrate against the new prompt style - re-run
# scripts/analyze_grounding_calibration.py against natural-output-style
# answers first if false-hedges on correct answers become a real problem,
# rather than guessing a new number.
ANSWER_CACHE_MIN_GROUNDING = 0.7

# Semantic cache: fuzzy match on query embedding similarity, checked before
# retrieval - a hit skips retrieval AND generation, not just generation like
# the literal cache. See scripts/calibrate_semantic_cache_threshold.py.
SEMANTIC_CACHE_MAX_SIZE = 1000
SEMANTIC_CACHE_TTL_SECONDS = 86400  # matches ANSWER_CACHE_TTL_SECONDS, same staleness reasoning
# Calibrated against 39 pairs (true-duplicate / related-but-distinct /
# unrelated) in this domain - true-duplicate and related-but-distinct
# scores do NOT separate cleanly (e.g. "gst registration process" vs "gst
# registration fees" scores 0.80, close to true duplicates). 0.92 sits
# above the highest observed related-but-distinct score (0.8963) with a
# ~0.024 safety margin, and below the threshold catches ~40% of true
# duplicates in the calibration set. Biased conservative on purpose: a
# missed semantic-cache opportunity costs ~220ms, a false match serves a
# wrong answer. Re-run the calibration script once real traffic
# accumulates - this was seeded, not measured against production queries.
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.92

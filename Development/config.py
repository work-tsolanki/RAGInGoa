import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "mock")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "mock")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "mock")
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

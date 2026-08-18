FROM python:3.11-slim

# Without this, Python fully buffers stdout in a non-TTY container, so
# print()-based startup logs (main_app.py's "Loading embedding model...",
# etc.) don't appear until the buffer flushes - making a slow step look like
# a multi-minute silent gap in `fly logs` instead of showing where time is
# actually going. Confirmed this exact confusion during cold-start timing
# tests against the live deploy.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# llama-cpp-python is deliberately excluded from the production image: it
# needs a C++ toolchain (or CUDA) to build and this deploy target has no
# GPU, so local generation is dropped in favor of Groq -> Claude (see
# GENERATION_BACKEND_ORDER_OVERRIDE in fly.toml). This is safe without any
# code change - GenerationService only imports llama_cpp when the GGUF
# model file exists on disk (src/generation_service.py), and that file is
# never copied into this image.
COPY requirements.txt .
# torch's default PyPI wheel bundles CUDA runtime libraries (nvidia-cublas,
# cudnn, nccl, etc. - several GB) even though this container has no GPU.
# Installing torch/torchvision from PyTorch's CPU-only index first means pip
# resolves everything else against those already-satisfied CPU wheels instead
# of pulling the CUDA build in as a transitive dependency.
RUN grep -v "^llama-cpp-python" requirements.txt > requirements.docker.txt \
    && pip install --no-cache-dir torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.docker.txt

COPY . .

# Pre-download both models used at request time instead of first-request
# time. Without this, /root/.cache is empty on every fresh container (it's
# not on the persistent volume), so every cold start - including any crash-
# restart while this URL is publicly live for judging - re-downloads them
# from Hugging Face before the app is ready, costing minutes. Baking them in
# makes that a one-time build cost instead of a recurring runtime one.
# Must match EMBEDDING_MODEL in config.py and CROSS_ENCODER_MODEL in
# src/guardrails.py exactly - a second, unbaked model (the cross-encoder)
# was missed on the first attempt at this fix and still downloaded live in
# production, which is exactly the kind of thing this comment is here to
# prevent recurring.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu'); \
    CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', device='cpu')"

EXPOSE 8000

CMD ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8000"]

"""Download the quantized local LLM used by GenerationService.

Model is gitignored (models/*.gguf, ~4.9GB) - run this once after cloning
to reproduce it locally.
"""

import os

import requests

MODEL_URL = (
    "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/"
    "resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)
OUT_PATH = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"


def main():
    os.makedirs("models", exist_ok=True)
    if os.path.exists(OUT_PATH):
        print(f"Already downloaded: {OUT_PATH}")
        return

    print(f"Downloading {MODEL_URL} -> {OUT_PATH} (~4.9GB)...")
    with requests.get(MODEL_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(OUT_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()

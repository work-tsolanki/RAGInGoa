# Local GPU Setup (Windows, RTX 50-series / Blackwell)

Enables GPU acceleration for embeddings, cross-encoder guardrails, and local LLM
generation via llama.cpp. Verified on an RTX 5070 Ti (compute capability sm_120).

## 1. Torch

The stable PyPI `torch` wheels don't include Blackwell (sm_120) kernels. Install
from the cu128 index instead:

```
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1+cu128
```

Verify with:
```python
import torch
torch.cuda.is_available()  # True
x = torch.randn(10, 10, device="cuda") @ torch.randn(10, 10, device="cuda")  # must not raise
```
`torch.cuda.is_available()` returning `True` is not sufficient proof - it only
checks the driver, not whether the installed build has kernels for this GPU's
compute capability. Always confirm with an actual op.

`embedding_service.py` and `guardrails.py` auto-detect CUDA and fall back to
CPU, so this step alone is enough for embeddings + grounding checks.

## 2. llama.cpp with CUDA (for local LLM generation)

The prebuilt `llama-cpp-python` wheel is CPU-only. Building with CUDA support
requires:

- **MSVC + CMake** - come bundled with Visual Studio (Community edition is
  fine). Confirm `cl.exe` and `cmake.exe` exist under
  `C:\Program Files\Microsoft Visual Studio\<ver>\...`.
- **CUDA Toolkit** (nvcc) - not bundled with the driver, install separately:
  ```
  winget install --id Nvidia.CUDA
  ```

Then build:
```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<ver>"
$env:CUDA_PATH_V<ver_underscored> = $env:CUDA_PATH   # e.g. CUDA_PATH_V13_3
$env:CUDAToolkit_ROOT = $env:CUDA_PATH
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:CMAKE_ARGS = "-DGGML_CUDA=on -DCUDAToolkit_ROOT=`"$env:CUDA_PATH`""
$env:FORCE_CMAKE = "1"
pip install llama-cpp-python --force-reinstall --no-cache-dir --no-deps
```

Gotchas hit along the way:
- **`CUDA_PATH_V<ver>` must be set explicitly**, even though the installer
  sets it at the machine level - a shell opened *before* the CUDA install
  won't see it (Windows doesn't retroactively inject env vars into running
  processes). Open a fresh shell, or set it manually as above.
- **CUDA 13.x moved the runtime DLLs** (`cublas64_13.dll`, `cudart64_13.dll`,
  etc.) from `CUDA_PATH\bin` to `CUDA_PATH\bin\x64`. `llama-cpp-python`'s own
  DLL-search fallback (`_ctypes_extensions.py`) only checks `CUDA_PATH\bin`
  and `CUDA_PATH\lib`, not `bin\x64`, so the built wheel fails to import with
  a generic `Could not find module ... (or one of its dependencies)` unless
  something puts `bin\x64` on the search path.
- **That fix must go through `PATH`, not `os.add_dll_directory()`.**
  `llama-cpp-python` loads its DLLs via
  `ctypes.CDLL(path, winmode=ctypes.RTLD_GLOBAL)`. Passing an explicit
  `winmode` makes the Windows loader use the classic DLL search order and
  ignore `os.add_dll_directory()`-registered paths - `add_dll_directory` only
  gets honored under `LOAD_LIBRARY_SEARCH_*` flags, which explicit `winmode`
  bypasses. `PATH`, on the other hand, is still searched under the classic
  order, so it works. `generation_service.py` handles this automatically at
  import time (see the block at the top of the file) - no manual PATH setup
  needed at runtime, only at build time above.

Verify with:
```python
from llama_cpp import llama_cpp
llama_cpp.llama_supports_gpu_offload()  # True
```
and check for `ggml_cuda_init: found 1 CUDA devices` in the logs when
`GenerationService` loads the model.

## Result

With all layers offloaded (`LOCAL_LLM_N_GPU_LAYERS=-1` in `config.py`, the
default), generation for the 8B Q4_K_M model dropped from ~10.4s (CPU) to
~0.3s (GPU) on the RTX 5070 Ti.

## CPU-only deployments (e.g. the GCP VM)

No action needed - `torch`, `embedding_service.py`, `guardrails.py`, and
`generation_service.py` all auto-detect CUDA and fall back to CPU cleanly. A
plain `pip install torch==2.7.1` (no cu128 index) and the stock
`llama-cpp-python` wheel are correct there.

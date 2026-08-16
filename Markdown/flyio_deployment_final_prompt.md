# Task: Deploy to Fly.io, with security fix, secrets migration, and production config decisions

## Context

Final remaining item before HH Goa 2026 Task 2 submission (deadline Aug 22). System is otherwise complete and verified: retrieval, caching, multi-strategy chunking (858,768-doc index confirmed live), grounding gate, Groq→Claude generation fallback, full regression-tested. This prompt covers the last mile: fix the one open security issue, then deploy to Fly.io.

**Do the security fix first. Do not deploy before it's done — deploying with the current dashboard/key setup exposes `RAG_API_KEY` to anyone who can reach the public URL.**

---

## Part 1: Fix the RAG_API_KEY exposure model (required before any public deployment)

### Problem
`main_app.py` currently injects `RAG_API_KEY` directly into the dashboard HTML at request time. This is only safe because the server is bound to `127.0.0.1` — nobody who can't already reach the API can reach the dashboard either. That protection disappears the moment this is publicly reachable.

### Fix — implement this option (simplest, lowest-risk for remaining time):
Stop injecting the key into HTML. Have the dashboard prompt for the key client-side and hold it in memory only (a JS variable), never rendered into the page source, never in localStorage/sessionStorage (per this project's existing browser-storage constraints).

1. In `main_app.py`, find and remove whatever currently injects `RAG_API_KEY` into the served HTML (likely an f-string or Jinja template variable in the dashboard route).
2. In `dashboard/index.html`, add a simple prompt-on-load:

```javascript
let ragApiKey = null;

function ensureApiKey() {
  if (!ragApiKey) {
    ragApiKey = window.prompt("Enter API key:");
  }
  return ragApiKey;
}

// Wherever the dashboard currently sends requests to your API
// (fetch calls, WebSocket connection headers/query params), use ensureApiKey()
// to get the key at call time instead of reading it from an injected variable.
```

3. Update every fetch/WebSocket call in the dashboard to use `ensureApiKey()` rather than a previously-injected global.
4. Test: view page source on the deployed dashboard, confirm no `RAG_API_KEY` value appears anywhere in the raw HTML/JS delivered to the browser.

### Verify before moving to Part 2
- [ ] `curl` the dashboard route directly, grep the response for the key value — must return zero matches.
- [ ] Confirm the dashboard still functions end-to-end after the prompt-based key entry (test a real query through it).

---

## Part 2: Move all secrets to Fly's secret manager

Do not put any of these in `fly.toml` — that file is often committed to the repo. Use `flyctl secrets set`, which stores them encrypted and injects as environment variables at runtime.

```bash
fly secrets set GROQ_API_KEY="<value>"
fly secrets set SARVAM_API_KEY="<value>"
fly secrets set ANTHROPIC_API_KEY="<value>"
fly secrets set RAG_API_KEY="<value>"
# Include INCEPTION_API_KEY only if Mercury 2 is actually being kept live —
# otherwise skip it, no reason to carry an unused secret into production.
```

Confirm none of these appear anywhere in `fly.toml`, `Dockerfile`, or any committed file — `fly.toml` only references secret *names* implicitly through the app's own `os.environ.get()` calls, never values.

---

## Part 3: Production backend configuration decision

Local llama.cpp fallback requires a GPU. Fly's standard machines don't have one (GPU instances exist but are unnecessary cost/complexity for this deadline). Drop local from the production fallback chain:

In `config.py`, for the production environment specifically (use an env var to distinguish from local dev if that distinction doesn't already exist):
```python
GENERATION_BACKEND_ORDER = ["groq", "claude"]  # local removed for production — no GPU on Fly
```
Keep local in the fallback dict in code (harmless, unreachable) rather than deleting the function — no reason to remove working code this close to the deadline, just don't route to it in prod.

---

## Part 4: fly.toml configuration

Create or update `fly.toml` in the project root:

```toml
app = "your-app-name"  # choose a name, must be globally unique on Fly
primary_region = "bom"  # Mumbai — closest to Ahmedabad-based dev/testing and likely judge location; change if judges are known to be elsewhere

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8000  # match whatever port main_app.py actually binds to
  force_https = true
  auto_stop_machines = false   # IMPORTANT: keep false for the judging window —
                                 # prevents cold-start WebSocket failures
  auto_start_machines = true
  min_machines_running = 1     # always-on, no scale-to-zero during judging period

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 1024   # start here, bump if the embedding/cross-encoder models need more; check logs for OOM on first deploy
```

**`auto_stop_machines = false` and `min_machines_running = 1` are deliberate, not defaults** — this is the fix for the cold-start-during-judging risk discussed earlier. Do not let these get optimized back to scale-to-zero before Aug 22, even to save cost — the risk of a judge hitting a stalled WebSocket handshake outweighs a few dollars of always-on compute for a week.

---

## Part 5: Confirm/write the Dockerfile

If one doesn't already exist for this project, create it. Adjust base image and dependencies to match actual `requirements.txt`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Confirm this matches how main_app.py actually starts the server —
# check for uvicorn command, host binding, and port
CMD ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Critical check**: `main_app.py` currently binds to `127.0.0.1` (verified during the security audit as `uvicorn.run(..., host="127.0.0.1", ...)`). This must change to `0.0.0.0` for Fly (or any host) to route external traffic to it — `127.0.0.1` inside a container is unreachable from outside the container entirely. Find this line and change it, or confirm the Dockerfile's CMD overrides it correctly if `main_app.py` reads host from an environment variable.

---

## Part 6: CORS check

If the dashboard is served from the same Fly app/origin as the API (likely, given current single-service structure), CORS may not be an issue. Confirm:
- [ ] Dashboard and API are served from the same `fly.toml` app / same origin.
- [ ] If they end up on different origins for any reason, add explicit CORS middleware in `main_app.py` (FastAPI's `CORSMiddleware`) allowing the dashboard's origin — do not use `allow_origins=["*"]` given `RAG_API_KEY`-gated endpoints exist; scope it to the actual known origin.

---

## Part 7: Deploy and verify

```bash
fly launch --no-deploy   # generates initial config if fly.toml doesn't exist yet, review before proceeding
fly deploy
```

After deploy:
- [ ] `curl https://your-app-name.fly.dev/health` (or whatever health endpoint exists) — confirm 200.
- [ ] Open the dashboard at the public URL, confirm the key-prompt flow from Part 1 works.
- [ ] Run a real end-to-end voice query through the public URL (not localhost) — record it, confirm STT → retrieval → generation → grounding all work over the public deployment.
- [ ] Run a text query through `/query` via `curl` with the `RAG_API_KEY` header, confirm auth works correctly (both that valid key succeeds and invalid key is rejected).
- [ ] Re-run a small subset (5-10 queries) of the benchmark set against the live public URL — confirm latency is in a reasonable range versus local dev numbers; some increase from Mumbai-region network hop to Groq/Sarvam is expected and fine, a large unexpected jump is not.
- [ ] Check Fly logs (`fly logs`) for any startup errors, missing env vars, or OOM warnings — fix before considering this done.
- [ ] Confirm the machine stays running after 20+ minutes of no traffic (tests `min_machines_running=1` actually took effect) — do not let this silently fall back to sleep behavior.

## Final checklist before calling Item 4 done

- [ ] Part 1 security fix verified (key not in page source)
- [ ] All secrets in Fly's secret manager, none in committed files
- [ ] `GENERATION_BACKEND_ORDER` correctly excludes local in production
- [ ] `min_machines_running = 1`, `auto_stop_machines = false` confirmed in live `fly.toml`
- [ ] Host binding changed from `127.0.0.1` to `0.0.0.0`
- [ ] Full end-to-end voice test passed against the public URL, not localhost
- [ ] `GITHUB repo` public link and this new live Fly URL both ready for the submission form

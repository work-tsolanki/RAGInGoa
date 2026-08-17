# Implementation: Switch to Neysa AI for India-based Inference

## Goal
Replace Groq with Neysa AI + Pipeshift inference (deployed in India, Llama 3.1-8B). Expected generation latency: 50-80ms (vs Groq's 175-270ms). No network RTT to US.

## Timeline
2-3 hours setup + test. If it works, commit and move to production. If it fails by end of Day 1, rollback to Groq.

---

## Step 1: Sign up to Neysa AI (15 min)

1. Go to https://neysa.ai (or https://pipeshift.io if Neysa redirects)
2. Sign up with your email
3. Verify email
4. Create a new project/workspace (name it "RAG-Goa" or similar)
5. Wait for onboarding email with API credentials

Expected: API Key + endpoint URL in your dashboard

---

## Step 2: Create a Llama 3.1-8B endpoint (30 min)

1. In Neysa dashboard, go to "Inference Endpoints" or "Models"
2. Select or search for `meta-llama/Llama-3.1-8b-instruct`
3. Create new endpoint:
   - Region: **India** (ap-south-1 Mumbai or ap-south-2 Hyderabad — choose Mumbai for Goa proximity)
   - Instance type: start with the default GPU instance (H100 or A100)
   - Replicas: 1
   - Auto-scaling: disabled (for predictable latency)
4. Deploy — wait for "Ready" status (typically 5-10 min)
5. Copy the endpoint URL (format: `https://api.neysa.ai/v1/...` or similar)

---

## Step 3: Get API credentials

1. In Neysa dashboard, go to "API Keys" or "Settings"
2. Generate a new API key
3. Copy and store securely (you'll add this to environment variables)

---

## Step 4: Test the endpoint locally before production

Create a test script `test_neysa.py`:

```python
import requests
import json
import time

NEYSA_API_KEY = "your_api_key_here"
NEYSA_ENDPOINT = "https://api.neysa.ai/v1/chat/completions"  # adjust URL per Neysa docs

def test_neysa():
    headers = {
        "Authorization": f"Bearer {NEYSA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/Llama-3.1-8b-instruct",
        "messages": [
            {"role": "user", "content": "What is a corporation?"}
        ],
        "max_tokens": 100,
        "temperature": 0.3,
        "stream": False
    }
    
    t0 = time.perf_counter()
    response = requests.post(NEYSA_ENDPOINT, headers=headers, json=payload)
    t1 = time.perf_counter()
    
    if response.status_code == 200:
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        latency_ms = (t1 - t0) * 1000
        print(f"✓ Success: {latency_ms:.1f}ms")
        print(f"Answer: {answer[:100]}")
        return latency_ms
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    latencies = []
    for i in range(5):
        lat = test_neysa()
        if lat:
            latencies.append(lat)
        time.sleep(1)
    
    if latencies:
        print(f"\nP50: {sorted(latencies)[len(latencies)//2]:.1f}ms")
        print(f"P100: {max(latencies):.1f}ms")
```

Run: `python test_neysa.py`

Expected output: latency in 50-150ms range (if lower, even better). If > 500ms, there's a problem.

---

## Step 5: Add Neysa to the fallback chain

In `generation_service.py`, add:

```python
import os
import requests
from openai import OpenAI

NEYSA_API_KEY = os.environ.get("NEYSA_API_KEY")
NEYSA_ENDPOINT = "https://api.neysa.ai/v1/chat/completions"  # verify exact URL from dashboard

def generate_neysa(prompt: str, max_tokens: int = None, temperature: float = None):
    """Stream generation from Neysa AI (India-based Llama 3.1-8B)"""
    if not NEYSA_API_KEY:
        raise RuntimeError("NEYSA_API_KEY not configured")
    
    headers = {
        "Authorization": f"Bearer {NEYSA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/Llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens or config.MERCURY_MAX_TOKENS,
        "temperature": temperature or config.MERCURY_TEMPERATURE,
        "stream": True
    }
    
    try:
        response = requests.post(NEYSA_ENDPOINT, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8").strip()
                if line.startswith("data:"):
                    try:
                        chunk = json.loads(line[5:].strip())
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        raise RuntimeError(f"Neysa API error: {e}")
```

Add to fallback chain in `generate_with_fallback`:

```python
BACKENDS = {
    "neysa": generate_neysa,
    "groq": generate_groq,
    "local": generate_local,
    "claude": generate_claude,
}
```

Update `config.py`:

```python
GENERATION_BACKEND_ORDER = ["neysa", "groq", "local", "claude"]  # Neysa is now primary
NEYSA_API_KEY = os.environ.get("NEYSA_API_KEY")
```

---

## Step 6: Deploy to production

1. Add `NEYSA_API_KEY` to your Fly.io secrets (or Render if you're using GPU tier):
   ```bash
   fly secrets set NEYSA_API_KEY="your_api_key"
   # or for Render:
   # Add via dashboard → Environment
   ```

2. Deploy:
   ```bash
   git push  # auto-deploys if you have CI/CD
   # or
   fly deploy
   ```

3. Test live endpoint:
   ```bash
   curl -H "RAG_API_KEY: your_key" \
     https://your-app.fly.dev/query \
     -d '{"query":"What is GST?"}' \
     -H "Content-Type: application/json"
   ```

4. Check latency via `/metrics` endpoint — confirm `generation_check` is now 50-100ms range

---

## Step 7: Measure full pipeline on live

Run the 40-query benchmark against the live Neysa endpoint:

```bash
python benchmark/run_backend_comparison.py
```

Expected results:
- Retrieval: ~40ms (unchanged)
- Generation: 50-150ms (vs Groq's 175-270ms)
- Grounding: 874ms (unchanged, still CPU-bound)
- **Total: ~950-1000ms** (improvement visible, but grounding still dominates)

If generation is > 200ms, something is wrong — check Neysa dashboard logs.

---

## Step 8: Rollback plan (if it fails)

If Neysa latency is not as expected or endpoint is unstable:

1. Revert config:
   ```python
   GENERATION_BACKEND_ORDER = ["groq", "local", "claude"]  # back to Groq primary
   ```

2. Redeploy:
   ```bash
   git push
   fly deploy
   ```

3. Verify `/metrics` shows Groq generation latency again

Total rollback time: ~5 min. No data loss, no state corruption.

---

## Critical notes

- **Endpoint URL**: Verify the exact API URL from Neysa's dashboard — it may differ from the example above
- **Streaming format**: Check Neysa docs for exact `stream=True` response format (should be standard OpenAI-compatible, but confirm)
- **Rate limits**: Neysa may have per-minute token limits on free tier — monitor logs if you see 429 errors
- **Auth header**: Verify whether it's `Bearer` or `X-API-Key` or another format — check Neysa docs

---

## Success criteria

✓ Neysa endpoint responds with generation latency 50-150ms
✓ Full pipeline latency drops from 1100ms to 950ms+ (grounding is still bottleneck, but generation is faster)
✓ All 40 benchmark queries complete without errors
✓ Fallback to Groq works if Neysa fails
✓ Production deployment is stable for 1+ hour

If all pass: commit, move to Phase 5 regression pass. If any fail: rollback and stay on Groq.

Timeline: finish by end of Day 1. If not working by evening of Day 1, roll back.

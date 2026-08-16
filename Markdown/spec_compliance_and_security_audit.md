# Task: Full compliance + security audit against HH Goa 2026 Task 2 spec

## Purpose

Cross-check the entire built system against the official task PDF, line by line, and separately audit the repository for exposed secrets. This is a verification pass, not a build task — the output should be a pass/fail/gap table per requirement, plus a clean-or-flagged security report. Do not "fix forward" into new features while auditing; log gaps first, then decide what's fixable before Aug 22.

## Part A: Requirements traceability — go through every line of the spec

For each item below: mark **PASS** (built and verified working), **PARTIAL** (built but incomplete or unverified), or **MISSING** (not built). Cite the actual file/function that implements it, or state plainly that none exists. Do not mark PASS on the basis of "this is planned" or "this should work" — only on verified, running code.

### 1. Pipeline shape: Voice input → Speech-to-text → Chunking/Retrieval → Answer generation
- [ ] Voice input capture (client-side audio recording) — check for this, likely **MISSING** based on session history.
- [ ] Speech-to-text integration — spec requires **Sarvam or ElevenLabs, pick one**. Check `requirements.txt`/`package.json` and `src/` for any STT integration. Based on the most recent confirmation ("currently voice layer is not added its purely a plain text"), this is expected to be **MISSING**. Confirm directly rather than assuming, and report exact status.
- [ ] Confirm architecture docs (from earlier project scaffolding — README/ARCHITECTURE.md) specified Sarvam — check whether that choice is still current or needs re-confirming given time remaining.
- [ ] End-to-end voice test: record a real audio clip, submit through the actual pipeline, confirm a spoken question produces a spoken or text answer through the full chain. If this test cannot currently be run at all, that's the single most important finding of this audit — report it first, above all other findings.

### 2. Chunking strategy — spec explicitly says "should be vast," rejects single naive fixed-size approach
- [ ] Identify every chunking strategy currently implemented (fixed-size, semantic, metadata-aware, overlap handling, etc.) — list them explicitly by name and file location.
- [ ] Confirm more than one strategy exists, or a clearly justified hybrid — a single fixed-size chunker does not satisfy this requirement as written.
- [ ] Confirm overlap handling is implemented, not just chunk boundaries.
- [ ] Confirm metadata is attached to chunks (source, position, language, etc.) and actually used somewhere in retrieval/ranking — metadata that's stored but never used doesn't satisfy "metadata-aware chunking" in spirit.
- [ ] If chunking strategy documentation exists (from earlier scaffolding work) but was never actually implemented in code, flag this explicitly — documentation is not implementation.

### 3. Latency target: full process (chunking + vector DB retrieval + everything through to final output) under 200ms
- [ ] **Resolve scope ambiguity first, before measuring anything**: does "full process ... through to final output" mean STT + retrieval + generation combined, or retrieval alone? Read the spec's pipeline diagram (Voice → STT → Retrieval → Generation) — the natural reading is the entire chain, since "final output" is the answer, not the retrieved chunks. Proceed on this interpretation unless you have direct clarification from organizers saying otherwise.
- [ ] Measure actual end-to-end latency under this full-chain interpretation: STT time (once implemented) + embedding + retrieval + generation + grounding, all summed, for a real spoken query.
- [ ] Compare against known numbers already gathered in this project: retrieval ~40ms, generation ~175-270ms (Groq) — these two alone may already exceed 200ms before STT time is even added. Report this honestly as a likely target miss under the full-chain interpretation, rather than reporting only the retrieval-alone number, which would misrepresent compliance.
- [ ] If the full-chain target cannot realistically be met by Aug 22, this needs to be a known, documented tradeoff going into submission — not a surprise discovered by a judge.

### 4. Latency analytics: P50 / P70 / P100, across a reasonable number of test queries, not a single best-case run
- [ ] Check whether `benchmark/run_backend_comparison.py` (or equivalent) currently records percentile statistics, or only per-query point measurements and averages. Based on session history, this harness currently reports per-query and average numbers, not percentiles — expected **MISSING or PARTIAL**.
- [ ] If missing: this needs a dedicated implementation — run a real query set (not the same 5-7 queries used throughout development; ideally 30-50+, sampled realistically) multiple times each, collect the full latency distribution, compute P50/P70/P100 (P100 = max, effectively worst-case), and produce this as a submittable artifact (table or chart), not just console output.
- [ ] Confirm the measurement includes full pipeline latency per the Part A.3 scope resolution, not retrieval-only.

### 5. Harness requirement: structured orchestration — tool calls, retries, structured I/O, error recovery — not a single raw prompt-in/text-out call
- [ ] Confirm `generate_with_fallback` (backend fallback chain: Groq → local → Claude) counts toward this — it does provide retries/error recovery across backends, which is a real point in favor.
- [ ] Check whether retrieval, generation, and guardrails are called as a structured pipeline with typed inputs/outputs at each stage (this appears to be the case based on session history — `StageTimer`, structured JSON logging, distinct function boundaries), or whether it's effectively one large function. If the architecture genuinely is modular (separate retrieval/generation/guardrail services, as described throughout this project), this likely satisfies the spirit of the requirement — confirm and document it explicitly for the submission writeup, since judges will be assessing this from the repo structure, not just runtime behavior.
- [ ] Confirm structured input/output typing (e.g. Pydantic models, typed dicts, or equivalent) exists at service boundaries, not just implicit dict-passing — this is the kind of detail a judge checking "real harness vs raw prompt-in/text-out" would look for directly in the code.

### 6. Guardrails: off-topic queries, unsafe/inappropriate inputs, hallucination checks, "knows when not to answer"
- [ ] Hallucination/grounding check: **confirm status** — `check_grounding` exists, but the threshold (`ANSWER_CACHE_MIN_GROUNDING`) was left uncalibrated as of recent sessions (0.0, effectively disabled as a gate). Verify whether calibration (Priority 2 from the master implementation prompt) was completed. If not, this requirement is only **PARTIAL** — the mechanism exists but doesn't actually gate anything yet, which is a meaningful gap given the spec specifically wants to see the system "know when not to answer."
- [ ] Off-topic query handling: **check for this explicitly** — nothing in the session history so far describes a dedicated off-topic detector (e.g., a classifier or similarity check against an out-of-scope reference set, as previously suggested as an optional routing layer). Based on available information, this is likely **MISSING** and needs to be built — even a lightweight version (embedding similarity against the corpus's topic centroid, or a simple keyword/domain classifier) is better than nothing here, since the spec calls this out by name.
- [ ] Unsafe/inappropriate input handling: **check for this explicitly** — confirm whether there's any input sanitization/rejection path distinct from the RAG-specific grounding check (e.g., handling for clearly abusive, harmful, or nonsensical input before it even reaches retrieval). Likely **MISSING**, needs a basic implementation.
- [ ] Confirm `validate_answer`'s current rejection logic (empty/too-short/refusal-phrase) is documented clearly as part of the guardrail story, since it does contribute to "knows when not to answer" even if narrower than the full spec ask.

## Part B: Security audit — hardcoded secrets and API key exposure

This is separate from Part A and should be run regardless of Part A's findings. Do this before any public GitHub repo link is submitted.

### B1. Scan git history, not just current files
Secrets committed once and later removed from the current file still exist in git history and are recoverable by anyone who clones the repo.

```bash
# Install and run gitleaks (or trufflehog as an alternative) against full history
pip install --break-system-packages detect-secrets  # or use gitleaks/trufflehog binaries if available in the network allowlist

# gitleaks (preferred if available):
gitleaks detect --source . --log-opts="--all" --report-path gitleaks-report.json

# trufflehog alternative:
trufflehog git file://. --json > trufflehog-report.json
```
If neither tool is installable in this environment, fall back to a manual grep pass across full history:
```bash
git log -p --all | grep -Ei "(api[_-]?key|secret|token|password|GROQ_API_KEY|INCEPTION_API_KEY|ANTHROPIC_API_KEY|SARVAM|ELEVENLABS)" 
```

### B2. Check every config/env file that might have been committed
```bash
git log --all --full-history -- "*.env" ".env*"
git ls-files | grep -iE "\.env$|\.env\.|config\.py$|secrets"
```
`.env` should never appear in `git ls-files` output — if it does, it was committed and needs history rewriting (`git filter-repo` or BFG Repo-Cleaner), not just deletion, since deletion alone leaves it in history.

### B3. Grep current codebase for hardcoded values, not just env-var references
Confirm every API key usage goes through `os.environ.get(...)` / `config.py`, never a literal string:
```bash
grep -rn "sk-\|gsk_\|AIza\|api_key\s*=\s*['\"]" --include="*.py" --include="*.js" --include="*.ts" .
```
Check specifically: `GROQ_API_KEY`, `INCEPTION_API_KEY` (Mercury), `ANTHROPIC_API_KEY` (Claude fallback), and whatever Sarvam/ElevenLabs key gets added once STT is implemented — audit these as a checklist, not just a general grep, since it's easy for one to slip through as a literal during fast iteration.

### B4. Confirm `.env` is gitignored, and was gitignored from the first commit
```bash
cat .gitignore | grep -i env
git log --all --diff-filter=A -- .gitignore  # check when .gitignore was first added
```
If `.gitignore` was added *after* the first few commits, check whether `.env` or config files were committed in that early window before the ignore rule existed.

### B5. Check for keys in non-obvious places
- Frontend/client-side code — if any API key is referenced in client-side JS (rather than proxied through your backend), it's exposed to anyone who opens browser dev tools, regardless of git history. Confirm all API calls to Groq/Mercury/Claude/STT providers happen server-side only.
- Jupyter notebooks, if any exist in the repo — notebook outputs can embed printed API keys from debugging sessions; check `.ipynb` files' raw JSON for output cells, not just source cells.
- CI/CD config files (GitHub Actions, etc.) if any exist — confirm secrets are referenced via GitHub Secrets, not inlined.
- README/documentation files — confirm no "here's my key for testing" leftover from setup instructions.

### B6. If anything is found
- Rotate the key immediately (regenerate from the provider's dashboard) — assume any committed key is compromised the moment it hits a public repo, even briefly, since bots scan public GitHub for exposed keys within minutes.
- Then remove from history with `git filter-repo` (preferred) or BFG Repo-Cleaner — plain `git rm` + commit is not sufficient, history remains.
- Force-push the cleaned history, and confirm no forks/clones of the compromised state exist that you don't control.

## Part C: Submission logistics cross-check

- [ ] GitHub repo link — confirm it's set to public (or accessible to judges per instructions) before the deadline, not just complete.
- [ ] Live working link — confirm actual deployed, reachable URL, tested from a clean browser session (not just localhost).
- [ ] Team/process video — 90 seconds, process not product.
- [ ] Demo video — end-to-end working demo. Given the STT gap in Part A.1, confirm this video does not imply voice functionality that isn't actually built — a demo showing capabilities not present in the live link is a credibility risk with judges.
- [ ] **Hashtag check — this task uses #RAGInGoa, not #FrameInGoa** (that's Task 1's tag). Confirm every post, every platform, every team member uses the correct tag for this submission.
- [ ] Confirm posts are on Instagram, X, and LinkedIn, by every individual team member (not one shared post), and at least one Instagram account is public, per spec.

## Output format

Produce a single markdown report: `AUDIT_REPORT.md`, with three sections mirroring Parts A/B/C above, each item marked PASS/PARTIAL/MISSING/CLEAN/FLAGGED with a one-line citation of evidence (file path, command output, or "not found — needs building"). Lead the report with the two highest-severity findings regardless of where they fall in the checklist order: STT pipeline status, and the 200ms full-chain latency scope resolution — these affect the most remaining work and should not be buried under lower-stakes items like hashtag correctness.

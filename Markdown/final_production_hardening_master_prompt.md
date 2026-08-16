# Master Prompt: Production Hardening & Submission Close-Out — HH Goa 2026 Task 2

## Context

Deployed and verified on Fly.io (`sin` region). Health, REST `/query`, WS `/ws/query`, auth, dashboard security fix, corpus integrity (858,768 docs), autostop/autostart config, and a 7-query live benchmark are all confirmed working on the public URL. Three real bugs were found and fixed with direct evidence during deployment (dependency drift, image-size pull failure, GGUF-triggered crash loop). This prompt covers everything remaining before the **Aug 22, 11:59 PM** submission deadline.

Work through phases in order. Each phase lists what to do and how to verify it. Do not skip verification steps to save time — every prior fix in this project that skipped direct verification (assumption instead of measurement) turned out to hide a real bug.

---

## Phase 1: Lock in tonight's fixes (do first, cheap, closes real risk)

1. **Pin the deployed image as the `fly.toml` default**:
   ```toml
   [build]
     image = "registry.fly.io/<app-name>:<the-exact-working-tag>"
   ```
   Verify: run `fly image show` (or equivalent) and confirm it matches what's actually serving traffic, not a guess.

2. **Clean up stray registry tags** (`data`, `data2`, `nomodel`, `deployment-*`, any others from tonight's iteration):
   ```bash
   fly image list  # or platform-equivalent
   # delete anything that isn't the pinned production tag
   ```
   Verify: only the pinned tag (and maybe one immediate-previous version for rollback) remains.

3. **Bake the embedding model into the image** (currently downloaded at runtime — costs 5-10+ min on every crash-restart, unacceptable given a live public URL a judge can hit anytime this week):
   - Add a `COPY` or `RUN` step in the Dockerfile to fetch/include the model at build time instead of first-request time.
   - Verify: kill the running machine (`fly machine restart <id>`), time how long until `/health` returns 200 again — should now be seconds, not minutes.

---

## Phase 2: Close the two open measurement questions

4. **Investigate the 13.9s P100 outlier** from the earlier percentile batch:
   - Pull the specific Gujarati query, its prompt length, which backend served it, and its timestamp relative to neighboring queries.
   - Check Groq/backend logs for any rate-limit or queueing signal at that timestamp.
   - Outcome: confirm it's genuinely a transient blip (document as such in the submission writeup with the specific evidence), or discover a real pattern (if so, this becomes a new fix item — don't silently absorb it either way).

5. **Manually review the 10 hedged answers** from the 25% hedge-rate batch:
   - Read each hedge alongside its query and the retrieved context that triggered it.
   - Confirm: hedges are on genuinely ambiguous/underspecified queries, not falsely triggered on answerable ones.
   - If any look like false hedges, this feeds directly into the Phase 3 temperature decision below — don't decide Phase 3 until this review is done.

---

## Phase 3: Grounding-gate consistency (judge-visible risk — a repeated question could hedge inconsistently)

6. **Decide, then implement, one of:**
   - **Lower generation temperature** (e.g. `0.3 → 0.1`) specifically to reduce phrasing variance and the resulting grounding-score jitter. Test: run the same query 5x in a row, confirm grounding score variance shrinks meaningfully versus the current setting, and confirm answer quality/naturalness (from the earlier prompt-style work) doesn't regress.
   - **Or**: leave temperature as-is and add an explicit line to the submission writeup documenting this as a known, disclosed limitation with the actual observed hedge-flip-rate as evidence you understand and measured it, not just hand-waved it.
   - Do not leave this undecided and undocumented — that's the only wrong outcome here.

---

## Phase 4: Remaining spec gaps

7. **Off-topic / unsafe input guardrail** (confirmed missing in the original audit; the spec names this explicitly: "handling for off-topic queries, unsafe/inappropriate inputs"):
   - Minimum viable version: embedding-similarity check of the incoming query against the corpus's overall topic centroid (or a small reference set of in-scope example queries) — reject/redirect if similarity falls below a threshold, before retrieval runs.
   - Add a basic unsafe-input check (obvious abusive/harmful input keyword or classifier gate) ahead of the RAG-specific grounding check — this is a different failure mode than "ungrounded answer," it's "shouldn't have been processed as a real query at all."
   - Test: run 5-10 deliberately off-topic queries (e.g. "write me a poem about cats," "what's 2+2") and 3-5 deliberately inappropriate ones, confirm the system declines gracefully rather than attempting retrieval/generation on them.
   - Wire this in as an early-exit stage in `main_app.py`, before the semantic/literal cache lookup — no reason to spend retrieval or generation cost on something that should be rejected immediately.

8. **Update `ARCHITECTURE.md`** to match what's actually implemented:
   - Chunking strategies (fixed-overlap + semantic-boundary, additive to base passage-level index)
   - Backend fallback chain (`Groq → Claude` in production, local excluded)
   - Grounding gate (now gates the returned answer, not just caching)
   - New off-topic/unsafe-input stage from item 7
   - Deployment target (Fly.io, `sin` region, volume-mounted corpus)
   - Verify: read the doc back against the actual code, confirm no stale claims remain (this exact gap — doc describing chunking that didn't exist — was caught once already during the audit; don't let a second version of that ship).

---

## Phase 5: Final full regression pass

9. Re-run the complete benchmark harness (English + Hindi + Gujarati set) against the live public URL, **after** all of Phases 1-4 land — confirm nothing regressed from the off-topic guardrail, temperature change, or model-baking change.
10. Re-confirm the four original deploy gates still hold: dashboard key exposure (zero matches), secrets (none in committed files), production backend order (local excluded), host binding (`0.0.0.0` in container).
11. Re-confirm `min_machines_running = 1` / `auto_stop_machines = false` are still live — check directly on the running machine config, not just `fly.toml` source, since this exact drift (config file vs. actual running state) happened once already tonight.

---

## Phase 6: Submission logistics

12. **Team/process video** (90 sec, process not product) — script and record.
13. **Demo video** — end-to-end, live public URL, real voice query shown working. Do not demo anything not actually live (e.g. don't show local-only fallback behavior if it's excluded in production).
14. **Post both videos**: Instagram, X, LinkedIn — every individual team member, every platform, tag **`#RAGInGoa`** exactly (not Task 1's `#FrameInGoa`). At least one Instagram account public.
15. **Fill the submission form**: GitHub repo link (public), Fly.io live URL, both videos. Confirm repo is actually public — a private repo at submission time is a silent, easily-missed failure.
16. **Submit** — no resubmissions allowed, so this is a one-shot action. Do a final read-through of every field before hitting submit.

---

## Non-negotiables throughout

- Every fix gets a direct verification step, not an assumption — this has been the pattern that caught all three real bugs tonight; don't relax it under end-of-week time pressure.
- Do not silently absorb a surprising result (Phase 2 items) without either explaining it with evidence or treating it as a new bug.
- Do not demo, document, or claim anything in videos/writeup/README that isn't actually true of the live deployed system as of submission.

## Suggested timeline against Aug 22, 11:59 PM deadline

| Day | Phases |
|---|---|
| Day 1 (tonight/tomorrow) | Phase 1, Phase 2 |
| Day 2 | Phase 3, start Phase 4 |
| Day 3 | Finish Phase 4, Phase 5 |
| Day 4 | Phase 6 — record videos |
| Day 5 | Post videos across all platforms/members, form draft |
| Day 6 (buffer, do not use as primary work day) | Final regression check, submit — with hours of margin before 11:59 PM, not a last-minute submission |

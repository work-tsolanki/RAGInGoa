# Post-deploy verification — do these in order

1. **Lock the fix**: add GGUF/model-weight patterns explicitly to `.dockerignore` (by name/extension) so this class of bug can't silently recur.

2. **Confirm data integrity**: verify the volume-mounted corpus is the full 858,768-doc reindexed version (check doc count via a query, not assumption) — not an earlier snapshot from the SFTP/Docker-push scramble.

3. **Confirm `min_machines_running = 1` / `auto_stop_machines = false`** are still live on the actual running app — several deploy attempts and region changes happened since this was set.

4. **Full regression pass on the live public URL** (not localhost):
   - End-to-end voice query: STT → retrieval → generation → grounding
   - 5-10 benchmark queries, compare latency vs local dev baseline — flag anything beyond expected network-hop increase
   - Dashboard key-prompt flow works, `curl` auth check (valid key succeeds, invalid rejected)
   - `fly logs` clean, no startup errors/OOM

5. **If all pass**: mark Item 4 done. GitHub repo link + this Fly URL go into the submission form.

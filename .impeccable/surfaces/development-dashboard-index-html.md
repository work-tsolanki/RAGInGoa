---
version: 1
slug: "development-dashboard-index-html"
primary_target: "Development/dashboard/index.html"
related_targets: []
---

Scope: Development/dashboard/index.html — the entire public-facing demo dashboard (single static page, no build step). Visitor mode: Persuade.

Audience, job, action, proof: hackathon judges/visitors at Hacker House Goa 2026, evaluating the voice-RAG system live. Job: ask a question (typed or spoken), read the answer and judge whether the system is trustworthy and fast. Action: type/speak a query; secondary action: expand the reconstruction log for technical scrutiny. Proof: live latency numbers, hybrid retrieval citations, and an explicit confidence/grounding score — never claims without the receipts.

Constraints: must preserve all existing functionality (WebSocket pipeline, live mic amplitude meter, streaming answer, developer log, theme toggle) and existing copy; must read as real engineering, not a consumer/marketing skin; must work fully on a judge's own phone, reachable without excess scrolling.

Chosen direction: brand-pinned rebuild on the official Hacker House Goa 2026 event brand kit (forest green / cream / hot pink / golden yellow, retro Goan travel-poster illustration) — see DESIGN.md for the full system and PRODUCT.md Brand Commitments for the source screenshots. This superseded an earlier unrelated "Collider Event Console" sci-fi direction from the same session; the brand screenshots are binding visual authority, not a free choice.

Memorable moment / signature motion: the verdict "stamp lands" sequence — hero sun-illustration pulses when a query fires, the chevron-ribbon phase tracker transitions smoothly through pipeline stages, and the CONFIRMED/UNCERTAIN/DECLINED verdict badge animates in like a rubber stamp hitting paper (oversized + rotated, snapping to a resting -2° tilt), with the rule drawing in and the confidence cells lighting up in a fast stagger. See DESIGN.md's Answer Verdict Panel component for the full spec and its Named Rule (The One-Stamp Rule) restricting this weight to the verdict moment only.

Unresolved: none material. This sandbox's browser viewport has repeatedly resized unreliably across sessions (reports success but doesn't always apply) — re-verify responsive/animation behavior with normal DevTools device emulation before the live demo if a cleaner environment is available.

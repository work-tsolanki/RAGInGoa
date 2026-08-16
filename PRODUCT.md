# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML/CSS/JS single-page dashboard (`Development/dashboard/index.html`) served against a Python FastAPI backend (`Development/main_app.py`), deployed on Fly.io.

## Users

Judges and visitors evaluating the project during the Hacker House Goa 2026 hackathon demo. They type or speak a question and expect to see, within seconds, whether the system retrieved and answered it correctly — this is a live, watched demo, not a private debugging tool.

## Product Purpose

A voice-enabled Retrieval-Augmented Generation (RAG) system: transcribes speech (Sarvam STT), retrieves relevant context via hybrid dense + BM25 search, and generates grounded answers with citations, confidence tier, and latency shown to the user. Success is a judge asking a question and immediately trusting the answer because the system visibly shows its work (sources, confidence, timing) rather than acting as an opaque chatbot.

## Positioning

Sub-200ms end-to-end voice-to-grounded-answer latency with visible grounding/confidence signaling and citation of retrieved sources — most RAG demos either hide latency and grounding entirely or are too slow to demo live.

## Operating Context

Live, on-stage/on-table hackathon demo: presenter or judge speaks or types a query on a laptop or phone, watches retrieval phases and a live audio meter, then reads the generated answer with confidence tier, cited documents, and a raw request/response log they can expand for technical scrutiny. Must hold up on mobile (recent work already went into mobile UX) since judges may view it on their own phones.

## Capabilities and Constraints

- Voice input via mic (Sarvam STT, language selector) and text input.
- Retrieval pipeline shows phase progress (STT → retrieval → generation → grounding).
- Answer states: grounded/confident, uncertain, and declined-to-answer (guardrail triggered) — each must read as visually distinct outcomes, not just color swaps.
- Confidence tier and per-document citation with relevance score shown per answer.
- Expandable raw log panel for technical/judge scrutiny.
- Connection/live status indicator (system may be online/offline).
- Sample "try it" query chips for fast demo starts.
- Latency and stats surfaced prominently (this is a claimed differentiator, not incidental).

## Brand Commitments

Name: "RAGInGoa" / "Hacker House Goa". The event runs an official brand kit — confirmed binding via reference screenshots at `Images/Screenshot 2026-08-16 080939.png`, `080945.png`, `080952.png`, `080958.png` (the official Hacker House Goa 2026 marketing site). Sampled directly from the assets:

- **Forest green** `#0B6839` — primary dark ground, alternates with cream as the section background.
- **Cream** `#FFFBE8` — light ground, alternates with green.
- **Hot pink / magenta** `#FF0080` — accent: category labels, chevron ribbons, dashed circle badges, the "गोवा" devanagari treatment inline in the wordmark.
- **Golden yellow** `#FEE101` — accent: primary CTAs, chevron ribbons, hero illustration (sun), badges.
- **Black** `#000000` — rare, used for a single status badge and a hairline divider bar.
- Bold poster-weight serif for display headlines ("HACKER HOUSE", "4 Days. One Rhythm.", "THE TIMELINE AT A GLANCE"), uppercase small-caps tracked mono for eyebrow labels and status badges ("PINNED UP", "RUNNING", "TO BE STARTED").
- Illustrated retro Goan-beach travel-poster grammar: hand-drawn line art (palm trees, sunset, beach houses, a scooter, a signpost), chevron/ribbon shapes for timeline and stat callouts, pill buttons with solid fills, dashed-circle badge framing, a repeating small-glyph divider pattern between sections.
- The wordmark treats "गोवा" (Devanagari) as an inline accent within the Latin "HACKER HOUSE GOA" lockup, set in the pink accent color at roughly cap-height — the dashboard's own "RAGInGoa" / "गोवा" treatment should follow this same pattern, not invent a different one.

This is the binding visual world for every surface of this project, including the dashboard — it is not a free aesthetic choice. Any dashboard-specific translation (functional states, data density, live telemetry) must be built from this system's actual materials, never from an invented unrelated palette.

## Evidence on Hand

Live backend at `Development/main_app.py` with real retrieval/generation/guardrail pipeline (see `Markdown/ARCHITECTURE.md`). No fabricated testimonials, pricing, or customer claims apply — this is a hackathon submission, not a commercial product.

## Product Principles

1. Trust is earned by showing work: latency, retrieval phases, confidence, and citations are core content, not chrome.
2. The demo must read correctly cold, in seconds, to someone who has never seen it — a judge, not a teammate.
3. Distinguish confident, uncertain, and declined answers unmistakably; a wrong-but-confident-looking answer undermines the positioning.
4. Must perform well live on both a presenter's laptop and a judge's own phone.

## Open Decisions

None. Visual world is now pinned to the official Hacker House Goa 2026 brand kit (see Brand Commitments) — resolved, not a free choice.

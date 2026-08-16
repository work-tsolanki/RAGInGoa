---
target: Development/dashboard/index.html
total_score: 26
max_score: 40
na_heuristics: 
p0_count: 2
p1_count: 2
timestamp: 2026-08-16T20-35-43Z
slug: development-dashboard-index-html
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 4 | Phase-pill tracker, connection badge, live mic-amplitude meter, streaming cursor, developer log stage timings are thorough |
| 2 | Match System / Real World | 3 | "Grounding score"/"grounding gate" jargon appears on the answer card unglossed |
| 3 | User Control and Freedom | 2 | No cancel/retry once a query is in flight or the socket drops |
| 4 | Consistency and Standards | 3 | `#how-btn` has no `aria-expanded`, while the structurally identical `#log-toggle` does |
| 5 | Error Prevention | 1 | Mic-permission denial (`getUserMedia` catch) produces zero on-screen feedback |
| 6 | Recognition Rather Than Recall | 4 | Chips, persistent asked-query/tier labels, visible phase state |
| 7 | Flexibility and Efficiency | 2 | No query history, no compare-against-prior-answer, no shortcut beyond click-to-record |
| 8 | Aesthetic and Minimalist Design | 3 | Undercut by the stats-grid overflow bug below |
| 9 | Error Recovery | 1 | WS error and backend-error paths render **no message at all** |
| 10 | Help and Documentation | 3 | Good "How this works" panel + log, but not cross-referenced with jargon in the answer card |
| **Total** | | **26/40** | **Acceptable** |

## Design Specificity Verdict

**LLM assessment**: Genuinely specific, not template-portable. The brand kit (forest green/cream/pink/yellow), the Rozha One + inline "गोवा" mixed-script wordmark matching the real event lockup, the chevron-ribbon phase tracker translating the event site's own timeline-ribbon graphic, the tri-state stamp reusing the site's actual "RUNNING"/"RESULTS OUT" badge colors, and the bilingual test-query chips are all load-bearing to this exact product and event. Nothing reads as a swappable skin.

**Deterministic scan**: `detect.mjs --json` ran in **DEGRADED** mode (regex fallback — htmlparser2/css-select/css-tree/domutils unavailable; no computed-contrast or DESIGN.md-token cross-check). 11 findings, all `advisory`/`quality`, all `design-system-font-size` / `design-system-radius` / `design-system-color` drift against the DESIGN.md ramp (7 font-size, 2 radius, 2 color — e.g. `index.html:69,113,147,242,250,254,259`, `:135,162`, `:213,221`). These are **low-signal / likely false positives**: the tool itself flags that it cannot verify against real tokens in degraded mode, and several (the fluid hero `clamp()`, mono footnote sizes) are legitimate intentional micro-typography, not drift.

**Visual overlays**: injection succeeded; the browser-side detector ran live in a **[Human]**-labeled tab and found 5 issues the CLI regex pass couldn't see:
1. **`div.xdivider` — tight-leading** (line-height 1.17, needs ≥1.3) — real but cosmetic, on a decorative "×" divider row.
2. **`.log-col-title` and `#log-no-run` — all-caps-body warnings** — both on currently-hidden elements (collapsed log panel), likely false positives pending visibility.
3. **`em` — low-contrast, 1.8:1 (needs 3:1)** — the pink "GOA 2026" in the footer title, `#ff0080` on `#0B6839`. **Real accessibility failure**, not a false positive — confirmed by computed RGB in Assessment B.
4. **`body` — cream-palette warning** — flagged as a generic "AI slop" cream background pattern. **This is a false positive by brief**: the cream (`#FFFBE8`) is the actual sampled color of the official Hacker House Goa event brand, not a generic default — the detector has no way to know the palette is brand-pinned.

## Overall Impression

The dashboard is doing something unusually hard well: it wears a real, specific, external brand convincingly, and its core "show the machinery" narrative (phases → verdict → citations) is clear and well-differentiated. The single biggest opportunity is that **the product's honesty story stops at the happy path** — the moment anything goes wrong (socket drops, mic permission denied, backend errors), the interface goes completely silent, which is the worst possible failure mode for a live, judged demo that's supposed to be *about* trustworthy, transparent behavior.

## What's Working

1. **The tri-state verdict is unmistakably distinct at a glance** — different border, rule pattern, stamp fill, and DECLINED's diagonal-stripe background, not a color swap on one template. This directly serves the product's core trust claim.
2. **The chevron-ribbon phase tracker is a functional reuse of the real event's own timeline-ribbon graphic** — a brand device repurposed for live pipeline state, not decoration bolted onto a generic progress bar.
3. **"गोवा" set inline in Rozha One matching the actual event wordmark's mixed-script treatment** — an easy-to-skip typographic detail that wasn't skipped, and it's the kind of thing a judge who's seen the real event branding would specifically notice and register as care.

## Priority Issues

**[P0] Silent failure paths during a live, judged demo**
- **Why it matters**: `ws.onerror = () => {}` (line 617), the `stage === "error"` handler (lines 642-647, which just resets phase to idle), and the mic `getUserMedia` catch block (lines 926-928, empty `catch (e) { return; }`) all fail with **zero user-facing message**. If the backend hiccups or a judge denies mic permission on their own phone, the page just... does nothing. In front of judges, this is indistinguishable from the demo being broken, and it directly undermines the product's own positioning ("trust is earned by showing work").
- **Fix**: Give every failure path a visible, on-brand message — reuse the DECLINED stamp/card grammar for a "connection lost" or "mic access needed" state rather than inventing a new one. At minimum, surface the WS close/error event as a message in the connection badge area.
- **Suggested command**: `/impeccable harden`

**[P0] Stats card overflows its own boundary at common desktop widths**
- **Why it matters**: **Both independent assessments found this bug, with slightly different numbers, confirming it's real and reproducible**: Assessment A measured the second stats column clipped at 891px viewport width; Assessment B independently measured `.stats-grid` children extending ~49.7px past `.stats-card`'s right edge at 875px. This isn't an edge case — it's an ordinary laptop or judge's tablet width, and it clips the "14 languages" / "0.70 grounding gate" numbers, which are the product's own differentiator stats.
- **Fix**: Audit `.stats-grid{grid-template-columns:1fr 1fr}` (line 109) and its parent `.stats-card` sizing — likely a `min-width:0` or `overflow` gap on the grid children, or the card's flex-basis not accounting for the `858,768`-digit column's real rendered width.
- **Suggested command**: `/impeccable optimize` (or `/impeccable adapt` if framed as a responsive-width issue)

**[P1] Footer "GOA 2026" fails contrast (1.8:1, needs 3:1)**
- **Why it matters**: Confirmed by computed RGB in Assessment B — the pink `em` (`#FF0080`) on the green footer band (`#0B6839`) measures 1.8:1, well under even the large-text WCAG AA floor of 3:1. This is a real accessibility failure on a branded, high-visibility element (the footer wordmark), not a false positive.
- **Fix**: Either lighten the pink specifically for this on-green context (the page already has a `--pink-ink` token used elsewhere for on-light-background text; a parallel on-green tint is needed here), or drop the pink accent on this specific instance and rely on weight/italics for emphasis instead.
- **Suggested command**: `/impeccable colorize` (contrast-focused) or fold into `/impeccable audit`

**[P1] Hero band loses its identity in night theme**
- **Why it matters**: `.hero-band{background:var(--green)}` (line 90) is hard-coded, but in dark theme the page's own ground is *also* `var(--green)` — so the hero band becomes visually indistinguishable from the page behind it. `.footer-band` correctly inverts for dark mode (line 212) but `.hero-band` has no equivalent override, breaking the "two bookend bands" symmetry DESIGN.md itself describes.
- **Fix**: Give `.hero-band` the same theme-aware override treatment `.footer-band` already has.
- **Suggested command**: `/impeccable adapt` or fold into `/impeccable polish`

**[P2] No feedback on ignored input, and no cancel/retry**
- **Why it matters**: Tapping a chip while a request is pending is a silent no-op (`submitQuery` early-returns on `pendingRequestActive`, line 838); tapping the mic before the socket connects does nothing (`toggleRecording` returns silently, line 921). Separately, once a request is in flight or the socket is down, there's no cancel or retry — only waiting or reloading. Assessment A's stress-tester persona (Riley) would plausibly conclude the page is broken rather than working-as-designed.
- **Fix**: Disable/visually gray chips and mic during a pending request rather than leaving them clickable-but-inert; add a lightweight cancel/retry control tied to the phase tracker.
- **Suggested command**: `/impeccable clarify`

## Persona Red Flags

**Jordan (first-timer)**: Lands on a "DISCONNECTED" badge with no context for expected wait time; if they click Speak before the socket opens, nothing happens and nothing explains why (ties to P0 above).

**Riley (stress-tester)**: Spam-clicking chips or the mic mid-request produces silent no-ops rather than any error or disabled affordance — could easily read as "this is broken" rather than "this is busy."

**Casey (mobile/judge's-own-device)**: Responsive layout held up well below 720px (hero stacks, phases scroll horizontally, copy button goes full-width) — but the stats-grid overflow bug reproduces at 875-891px, a width squarely plausible on a judge's tablet or a laptop browser that isn't maximized, not just a phone edge case.

## Minor Observations

- `div.xdivider`'s line-height (1.17) is tighter than typical body-text guidance (≥1.3) — cosmetic, low priority, since it's a decorative "×" row, not reading text.
- Zero-citation handling for a grounded/uncertain answer is indistinguishable from the intentional declined-with-no-docs case — consider an explicit "no supporting passages found" message.
- `#send-btn` starts `disabled` and was observed to be **skipped in natural Tab order** while disabled (Tab went straight from mic-btn to the language `<select>`) — expected browser behavior for a disabled control, but worth confirming the intended keyboard path once the socket is connected and the button becomes enabled.
- `#mic-btn`'s `:focus-visible` outline is present (verified via real Tab navigation, not just `.focus()`) but visually blends with the button's own permanent 2px pink border — a keyboard user may not perceive a distinct focus state there even though one exists.
- Urdu ("اردو Urdu") sits in a plain `<option>` with no bidi consideration — worth a manual check in a real browser, not verified here.
- `#answer-text` has no length clamp; a long ungrounded/rambling answer could push citations and the log far down the page.

## Questions to Consider

1. If the socket never connects before judges arrive, is a static "disconnected" dot with no retry button the failure mode you actually want them to see?
2. The stamp-land motion is the product's one authored moment, but it plays identically for CONFIRMED, UNCERTAIN, and DECLINED — should DECLINED get a different physical metaphor than the same "landing" motion used for a win?
3. Should the 96px hero headline compress or step back once a query has been answered, so the verdict — not the permanent brand headline — is what a judge's eye returns to on a second glance?

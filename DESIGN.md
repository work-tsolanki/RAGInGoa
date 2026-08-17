---
name: RAGInGoa — Hacker House Goa 2026 Brand
description: The dashboard rebuilt on the official Hacker House Goa 2026 brand kit — forest green, cream, hot pink, golden yellow, retro Goan travel-poster illustration.
colors:
  forest-green: "#0B6839"
  green-deep: "#084A29"
  cream: "#FFFBE8"
  cream-dim: "#FFF3CE"
  hot-pink: "#FF0080"
  pink-ink: "#C2005F"
  pink-on-band: "#FFC2E0"
  golden-yellow: "#FEE101"
  yellow-ink: "#7A6000"
  black: "#000000"
typography:
  display:
    fontFamily: "'Rozha One', Georgia, serif"
    fontSize: "clamp(2.5rem, 8.4vw, 6rem)"
    fontWeight: 400
    lineHeight: 0.94
    letterSpacing: "-0.01em"
  body:
    fontFamily: "'Sora', system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "'IBM Plex Mono', monospace"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "0.12em"
rounded:
  sm: "8px"
  md: "16px"
  lg: "22px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.golden-yellow}"
    textColor: "#0B3D22"
    rounded: "{rounded.pill}"
    padding: "10px 18px"
  button-outline-pink:
    backgroundColor: "{colors.cream}"
    textColor: "{colors.pink-ink}"
    rounded: "{rounded.pill}"
    padding: "10px 18px"
---

# Design System: RAGInGoa — Hacker House Goa 2026 Brand

## Overview

**Creative North Star: "The Official Brand, Working"**

This is not an invented visual world — it is the actual Hacker House Goa 2026 event brand kit (confirmed via reference screenshots of the official marketing site, see `PRODUCT.md` → Brand Commitments), applied to a live, functioning voice-RAG demo console. The brand is a retro Goan-beach travel-poster identity: forest green and cream sections, hot pink and golden yellow as equal-weight accents, hand-drawn line illustration, chevron/ribbon shapes, pill buttons, and a bold poster serif paired with tracked mono labels. The dashboard translates this system into working UI: a green hero band (mirroring the official homepage's hero) bookends a cream working area (query bar, phases, answer, log) and a green footer band, so a judge recognizes the actual event brand within the first second — not a designer's invented alternative to it.

**Key Characteristics:**
- Forest green and cream alternate as full-section grounds, exactly as the official site does — never mixed into a single muted "brand color"
- Hot pink and golden yellow are reserved, high-saturation accents — never diluted into pastel tints
- Rozha One (bold poster serif with real Devanagari support) carries every display headline, including the wordmark's "गोवा" — chosen specifically because the brand's own wordmark mixes Devanagari into a Latin display face, and Rozha One is the one available face that renders both scripts as one coherent type family
- IBM Plex Mono carries every label, badge, and data readout, echoing the brand's tracked small-caps eyebrow labels ("PINNED UP", "RUNNING")
- Chevron-ribbon shapes, dashed borders, and an "×" divider row are drawn directly from the brand's own timeline ribbons, dashed circle badges, and repeating pattern dividers
- Real, offset card shadows (not neumorphic, not zero-offset glow) — matching the official site's card elevation

## Colors

Four named roles, used at full saturation, matching the official brand kit exactly (values sampled directly from the reference assets).

### Primary
- **Golden Yellow** (`#FEE101`): the "confirmed/active" signal — primary CTA (send button), the confirmed-answer stamp/rule/confidence cells, live progress fills, the active phase-ribbon segment. Reuses the official site's own "RUNNING" badge color for "this is active/good."

### Secondary
- **Hot Pink** (`#FF0080`, text form `#C2005F`): the brand's general vivid accent — the wordmark's "गोवा", the mic button, category labels, the uncertain/hedged answer state, focus rings, selection color.

### Tertiary
- **Black** (`#000000`): reserved for exactly one meaning — a closed/declined state, reusing the official site's own black "RESULTS OUT" (closed) badge color. Never used decoratively.

### Neutral
- **Forest Green** (`#0B6839` / deep `#084A29`): primary ink and the hero/footer band background — this is the brand's dominant dark tone, not a muted gray.
- **Cream** (`#FFFBE8` / dim `#FFF3CE`): the light ground and every card surface, in both themes.
- **Muted Green** (`#4C6B57`, fixed, non-themed): secondary text on any cream/white card — deliberately not swapped by the theme toggle, since every card stays cream regardless of theme.

### Named Rules
**The Four-Role Rule.** Only green, cream, pink, and yellow (plus black for the one declined state) ever appear as fills. No fifth color, no tint/shade gradient of these four — the brand is bold and flat, not a designer palette extrapolated from it.

**The Card-Is-Always-Cream Rule.** Every card/panel surface is cream or white in both themes; only the *page ground* swaps between cream (day) and green (night). This mirrors the official site's own pattern of cream cards floating on a green section.

**The Bookend-Bands-Match Rule.** The hero band and footer band are one symmetric pair: both are green in day theme, both flip to cream in night theme, together — never independently. Raw `--pink` measures only ~1.8:1 on `--green` (fails the 3:1 large-text floor), so pink *text* set directly on a bookend band always uses `--pink-on-band` (a light rose in day theme, `--pink-ink` in night theme) instead of the raw accent; pink *fills* (buttons, ribbons, badges) are unaffected and keep the raw brand pink.

## Typography

**Display Font:** Rozha One (single weight; its natural weight is already poster-bold, so no synthetic bold/italic is ever applied)
**Body Font:** Sora (300–800)
**Label/Mono Font:** IBM Plex Mono (400–700)

**Character:** Rozha One's flared, moderately-bracketed serif carries the H1 and footer title, and — critically — the Devanagari "गोवा" in the same face as the Latin type, exactly matching the official wordmark's mixed-script treatment. Sora carries all prose. IBM Plex Mono carries every label, badge, stat number, and timestamp.

### Hierarchy
- **Display** (400, `clamp(2.5rem, 8.4vw, 6rem)`, 0.94): H1 headline and footer title only, uppercase.
- **Body** (400, 17px / `--fb`, 1.5; 21px / `--fl` for the primary answer paragraph): tagline, answer text, chip labels, how-this-works copy.
- **Label/Data** (500–700, 11–13px, letter-spacing 0.08–0.16em, uppercase, tabular numerals on live values): buttons, badges, stat numbers, phase-ribbon segments, log rows.

### Named Rules
**The No-Synthetic-Style Rule.** Rozha One has one real weight and no italic. Never set `font-weight` above 400 or `font-style: italic` on it — a fake-bold or fake-italic serif is a browser artifact, not a brand choice.

## Layout

Single-column wrapper (`1180px` max, centered, safe-area-aware gutters). A green **hero band** and a green **footer band** (both rounded, `22px` radius) bookend a cream working area. The hero is a two-column flex row: headline + tagline + brand ribbon on the left, a cream telemetry card on the right, with a small hand-authored sun/wave line illustration filling the gap between them (not hidden behind the card, not full-bleed wallpaper). An "×" divider row marks the seam between the hero band and the working area, echoing the brand's own repeating-pattern dividers.

Below 720px: bands lose their side illustration (no room to read at that scale), the phase-ribbon row scrolls horizontally rather than compressing five labels into unreadable slivers, and all interactive targets meet a 44px+ touch floor with 16px inputs. The query bar is positioned to be reachable without scrolling past the full pitch — asking a question is the page's job, not a reward for scrolling.

## Elevation & Depth

Real, offset card shadows — `0 10px 24px rgba(11,60,35,.14), 0 2px 4px rgba(11,60,35,.08)` in day mode, a darker equivalent in night mode — matching the official site's own card elevation. This is a deliberate departure from a flat/glow system: the brand's cards visibly lift off the page.

### Named Rules
**The Real-Shadow Rule.** Card shadows always carry a real offset and blur (never zero-offset). No glow, no neon halo, ever.

## Shapes

Generous, friendly radii (`8px`/`16px`/`22px`) on cards and bands — nothing angular or instrument-panel-like. Buttons, chips, badges, and the brand pill are always fully rounded (`999px`), matching the official site's pill buttons exactly. Dashed borders (`1.5px dashed`) mark secondary/framing elements — stat boxes, the recording mic button, citation rows — echoing the brand's dashed-circle badge motif.

The signature shape device is the **chevron ribbon**: a `clip-path` arrow-notch shape used for the phase tracker, directly translating the official site's timeline-ribbon graphic into a live pipeline-stage indicator.

### Named Rules
**The Chevron-Ribbon Rule.** Any sequential/stage-based UI (the phase tracker today; any future step indicator) uses the chevron-ribbon shape, never a plain segmented bar — it's the brand's own device for "sequence," reused functionally.

## Components

### Buttons
- **Shape:** fully pill (`999px`), 2px solid border.
- **Primary (`#send-btn`):** golden-yellow fill, dashed green border, dark-green icon — the brand's solid-fill CTA pattern.
- **Outline (`#mic-btn`, header buttons):** cream/white fill, pink or green 2px border; hover inverts to a solid pink fill — the brand's outline-button pattern (its "TASK DETAILS" links).
- **Recording state:** solid hot-pink fill, dashed border — an active/urgent variant of the outline button.

### Chips
- **Style (`.chip`):** pill, cream background, hairline border, Sora body text with an IBM Plex Mono language tag. Hover swaps border to pink.
- **Lifecycle:** the whole "Fire a test event" block (`#try-block`) hides the moment a query is submitted (voice or text) and does not return — once a visitor has asked something, the suggestions have done their job and stop competing with the answer for attention.

### Cards / Containers (`.panel`)
- **Corner Style:** `16px` radius.
- **Background:** always cream/white (`--panel`), regardless of theme.
- **Shadow Strategy:** real offset shadow — see Elevation & Depth.
- **Border:** none by default; dashed hairline on framing sub-elements (stat boxes, citation rows).

### Stat / Telemetry Card
- **Style:** 2×2 grid of dashed-border cells inside the cream hero card. Numbers in Rozha One; labels in IBM Plex Mono uppercase.

### Phase Ribbon Tracker (signature component)
A row of chevron-clipped segments (`.phase-pill`) reading the pipeline stage. Past segments: cream-dim fill. Current segment: solid golden-yellow, bold. Future segments: dimmed. Below 720px it becomes a horizontal-scroll strip rather than compressing. Segment color/opacity changes transition (220ms ease) rather than snapping, so progress through the pipeline reads as continuous motion.

### Answer Verdict Panel
Tri-state card reusing the brand's own status-badge colors: **CONFIRMED** (golden yellow — the brand's "active" color), **UNCERTAIN** (hot pink, dashed rule), **DECLINED** (black border, dashed black rule, diagonal cream-stripe background — the brand's "closed" color). A 12-cell confidence bar lit in the state color, a tabular grounding-score readout, and — when applicable — citation rows with the rank number set in Rozha One italic-free serif and a dashed border frame.

**Signature motion: the stamp lands.** When an answer resolves, the verdict badge animates in like a rubber stamp hitting cream paper — oversized and rotated, snapping down to a resting -2° tilt in 480ms (`cubic-bezier(0.16,1,0.3,1)`), reusing the brand's own physical stamp/badge language rather than a generic fade. The rule beneath it draws left-to-right (460ms) as the stamp settles, the confidence cells light up left-to-right in a fast stagger (22ms/cell), and citation rows rise in with a short stagger (70ms apart). This is the one authored focal sequence in the product — query fires (hero sun pulses) → pipeline runs (phase ribbon transitions) → verdict stamps down (this sequence) — and it is not to be diluted into a plain fade on future states.

**The answer itself is highlighted**, not just the badge: `#answer-text` carries a soft background tint in the same state color as the stamp/rule/cells (yellow ~28% for confirmed, pink ~10% for uncertain, black ~5% for declined), fading in ~160ms after the text lands so the highlight reads as a reveal, not a static box. This threads the state color through every signal on the card — stamp, rule, confidence cells, and now the answer text itself — so confidence is legible even at a glance.

### Named Rules (motion)
**The One-Stamp Rule.** The stamp-land motion is reserved for the verdict badge landing. Do not reuse it for routine UI feedback (buttons, toggles) — those get fast, quiet transitions instead, so the stamp keeps its authored weight.

### Corpus Note
A short, plain-language paragraph beneath the phase tracker, written for a non-technical visitor, not a fellow engineer: "a general-knowledge lookup, not a search engine," with two concrete example questions and a one-line explanation that an uncertain/no-answer result means the system is being honest, not broken. No dataset name, no ML terminology (earlier drafts named MSMARCO-XI and "open-domain" — cut for a normal reader). Persists after the "Fire a test event" chips are dismissed, since it answers the same "what can I ask" question the chips do, without needing them present. Body copy (Sora), not a label — a full sentence read at conversational size, never mono/uppercase.

### System Notice (signature component)
The one visible surface for failures that would otherwise happen silently: a dropped/unreachable connection (with automatic backoff reconnect and a manual "Retry now"), denied or missing microphone access, and a backend pipeline error mid-query. Styled with the same black/dashed/diagonal-stripe language as the DECLINED answer state — reusing the brand's "closed" semantic — but rendered as its own banner near the query bar, never inside the answer card itself, so a system failure is never mistaken for an answered outcome. Connection loss auto-hides once reconnected; permission and pipeline errors carry a dismiss control.

### Named Rules (system notice)
**The Silence Ban.** No failure path (socket error, mic permission denial, pipeline error) may resolve with zero on-screen message. Every one routes through the System Notice component.

## Do's and Don'ts

### Do:
- **Do** keep green and cream as full-section (not accent) grounds — this is how the official brand actually alternates its sections.
- **Do** keep pink and yellow at full saturation; never tint them toward pastel.
- **Do** use Rozha One at its native weight only — no synthetic bold/italic.
- **Do** keep every card cream/white regardless of theme; only the page ground swaps.
- **Do** give a grid/flex container `min-width:0` (or `minmax(0,1fr)` columns) whenever its content — especially a large tabular number — could force it wider than its track; the stats grid overflowed its card until this was added.
- **Do** route every failure path through the System Notice component (see The Silence Ban).

### Don't:
- **Don't** invent a fifth accent color or a gradient between the four brand colors.
- **Don't** use a zero-offset glow shadow anywhere — the brand's own cards use real, offset elevation.
- **Don't** set raw `--pink` as text color directly on a green or cream band — use `--pink-on-band` (see The Bookend-Bands-Match Rule).
- **Don't** put a kicker or eyebrow label above a heading; let the display serif carry the weight.
- **Don't** reuse black for anything but the declined/closed state.

# OpenClaw Voice — UI Research: Hands-Free / In-Car Voice UX

**Date:** 2026-07-07 · **Phase 4 deliverable** · feeds PRD §6 (UI/UX principles) and Milestone M3
**Method:** Web research across official docs, help centers, launch posts, hands-on reviews, and SDK source (July 2026 state). Claims marked verified cite a source; inferred = reasonable reading of thin sourcing.
**Screenshots:** `docs/ui-research/screenshots/` is empty this pass — the surveyed UIs are mobile-app surfaces not capturable from a desktop browser session, and press images are copyrighted. Each product section links a source with imagery instead; grab in-app screenshots from Nathan's phone during M3 design review if wanted.

---

## Part 1 — Product survey

### 1. ChatGPT Voice Mode (Advanced Voice) — OpenAI
- **Turn detection:** server-side VAD inside the speech-to-speech model; no wake word; no published silence timeout. Reviewers call it aggressive — coughs/pauses/background noise read as turn-ends or interruptions.
- **Interruption:** yes, talk-over works; known bug where tap-to-interrupt froze input. Full-duplex reportedly coming.
- **Visual states:** the big 2025–26 story — OpenAI **abandoned the full-screen blue orb** as default (Nov 2025) and made voice a layer inside the normal chat: live transcript forms as you talk, rich cards (maps/photos) render alongside; legacy "Separate Mode" orb still optional. Orb behavior: pulse = listening, morph = speaking.
- **Kill switch:** mute (bottom-left mic) + end (bottom-right X). No red-phone metaphor.
- **Latency:** published 232 ms min / 320 ms avg time-to-audio (GPT-4o launch post) — the class benchmark.
- **Hands-free/driving:** background audio works; no native CarPlay UI (audio-through-car-speakers only). Transcript persists to chat history — good pattern.
- **Take:** imitate transcript-persistence and voice-as-a-layer; avoid their oversensitive endpointing.
- Sources: help.openai.com voice FAQ · openai.com/index/hello-gpt-4o · phonearena.com (in-chat voice) · community.openai.com (interrupt bug)

### 2. Grok Voice (xAI)
- **Turn detection:** full-duplex single model (listen/reason/speak concurrently); **no user control over endpointing** — silence commits the turn, users report feeling rushed and invent protocol hacks ("I'll say 'over'"). Top HN complaint.
- **Interruption:** best in class — adjusts mid-stream instead of resetting; but self-interrupts in noisy rooms (echo/noise → model hears "speech").
- **Visual states:** minimal; live caption of your speech; personality/voice picker. Voice auto-starts in some contexts (documented user annoyance — anti-pattern for us: never grab the mic unexpectedly, arm explicitly).
- **Kill switch:** tap-out; thinly documented.
- **Latency:** xAI claims <1 s avg time-to-first-audio, "5× faster than nearest competitor"; independent τ-voice bench gave it 73.7% task completion under phone-line conditions vs ~21% for ChatGPT/Gemini Live.
- **Take:** proof that latency + real barge-in is the core feel of a great voice product; also proof that **no silence-tolerance control = the #1 user complaint** — our 15–20 s patience requirement is exactly the gap.
- Sources: docs.x.ai voice agent · x.ai Grok Voice Think Fast 1.0 · news.ycombinator.com/item?id=44518555

### 3. Google Gemini Live
- **Turn detection:** automatic VAD with context-aware pause handling (a 2026 review noted it waits "the appropriate amount of time expected in human conversation"); wake phrase "Hey Google, let's talk Live"; no published numbers.
- **Interruption:** speak to interrupt, **or disable interruptions in settings** — the only consumer product with user-controllable barge-in. Tap-screen alternate interrupt.
- **Visual states (April 2026 redesign):** fullscreen replaced by a **floating pill overlay** — blue waveform center, screen-share/keyboard buttons flanking, captions toggle; **condenses to a small circle** when you use other apps; app-edge multicolor glow = listening identity; in-session control bar has explicit **red End button**.
- **Kill switch vocabulary — richest surveyed:** **Hold** (pause, mic muted, session alive) vs **Mute** (mic off) vs **End** (session closes, transcript surfaces). Swipe-right also ends.
- **Latency:** "sub-300 ms" marketing for the Live API; real-world dev reports of 2–6 s under load — marketing ≠ field truth.
- **Hands-free/driving:** best surveyed — session continues on lock screen, background mode under other apps, native Gemini in Android Auto (Assistant fully retired there March 2026).
- **Take:** steal Hold/Mute/End as three distinct verbs, the shrink-to-circle overlay, and post-session transcript.
- Sources: support.google.com/gemini/answer/15274899 · 9to5google.com Gemini Live redesign (Apr 2026) · android.com Gemini on Android Auto

### 4. Meta AI Voice
- **Turn detection:** half-duplex VAD default; opt-in labeled "full-duplex demo" toggle (adds overlap + audio ducking, disables some features while on — honest experimental staging).
- **Visual states:** waveform icon → circle view; persistent **"AI is listening" floating banner** — the cleanest "am I being listened to?" trust affordance surveyed. State choreography otherwise thin.
- **Kill switch:** poorly documented (toggle-based). **Latency:** ~250 ms claimed in full-duplex mode.
- **Driving:** nothing native; hardware story is Ray-Ban glasses hand-off.
- **Take:** copy the always-visible listening banner concept (ours: live mic-energy motion, which is self-verifying).
- Sources: meta.com help AI voice pages · datastudios.org Meta AI voice mode review

### 5. Perplexity Voice
- **Turn detection:** VAD + follow-up-ready listening; **push-to-talk offered as a first-class settings toggle** (unique among consumer apps); "Hey! Plex" wake word on Samsung.
- **Interruption:** voice barge-in on iOS.
- **Visual states:** reactive "sphere of dots" while listening; live transcript toggle; gear for voice/PTT settings.
- **Kill switch:** mute option; **session keeps running in background until you hit X — documented foot-gun** (forgotten hot mic).
- **Latency:** inherits OpenAI Realtime stack in Comet; no first-party number.
- **Driving:** deepest OS integration — can be Android's default assistant (power-button invoke, overlay over apps).
- **Take:** keep PTT as a first-class *option* (matches our desktop story 12); never let a session outlive the visible UI.
- Sources: perplexity.ai help center (iOS/Android assistant) · techpp.com icons explained · androidpolice.com

### 6. Siri hands-free / CarPlay
- **Turn detection:** wake word + steering-wheel button; **hold-to-talk explicit endpointing** (keep button held, release to commit) as an override when VAD would guess wrong — decades-tested pattern.
- **Interruption:** no voice barge-in; button re-press cancels.
- **Visual states:** iOS 18+ full-edge glow (listening) replacing the orb; in CarPlay Siri is a **bottom overlay, never a full-screen takeover** — the map stays visible. Captions/"Always Show Request" for voice-only error recovery.
- **Driving canon (NHTSA Phase 1 guidelines):** single glances ≤ 2 s, cumulative ≤ 12 s per task; voice encouraged precisely to keep eyes on road. This is the regulatory basis for our "peripheral-legibility" principle.
- **Take:** overlay-not-takeover; captions confirm what was heard; hold-to-talk as the manual escape hatch.
- Sources: support.apple.com Siri-in-car guide · federalregister.gov 2013-09883 (NHTSA guidelines)

### 7. Google Assistant driving mode (deprecated) → lessons
- Driving Mode was stripped (Feb 2024) then killed in the Gemini transition; Assistant removed from Android Auto March 2026. Hands-on reviews found Gemini **slower than Assistant for command-style tasks** — an explicit warning: LLM flexibility must not cost the ≤2 s feel on simple commands ("what time is it?" must not wait out a patience window — semantic end-of-turn matters).
- Sources: androidcentral.com driving mode shutdown · 9to5google.com Gemini-on-Auto hands-on

### 8. Rabbit R1 — post-mortem
- Pure push-to-talk, no VAD; animated rabbit (ears perk when listening) but **no distinct thinking state**; measured 11 s query latency; silent failures (asked traffic, got weather or nothing).
- **Lessons:** (1) latency without visible state = users can't tell "thinking" from "dead" and re-press, self-interrupting; (2) silent failure kills trust faster than wrong answers; (3) PTT demands a free hand — disqualifying for driving.
- Sources: engadget.com R1 review · tomsguide.com R1 review · gizmodo.com R1 review

### 9. Humane AI Pin — post-mortem
- No wake word (privacy-by-design tap-to-talk); Trust Light LED as hardware listening indicator; ~6 s answers vs near-instant Siri/Alexa; palm-laser display unreadable in sun; battery/thermals killed always-ready ambition; servers bricked Feb 2025.
- **Lessons:** voice-only fails tasks needing scan/compare/re-read (our transcript pane earns its place); the Trust Light validates a dedicated, hardware-honest "mic is live" indicator; latency + missing basics broke trust irreversibly.
- Sources: engadget.com Pin review · spyglass.org brutal-review roundup · failure.museum/humane-ai-pin

### 10. Character.AI voice — call-screen metaphor; **tap**-to-interrupt (avoids false voice barge-in, costs hands-free purity); transcript renders live into the chat. End-call button. (blog.character.ai Character Calls)

### 11. Pi (Inflection) — best-in-class TTS prosody undone by **~0.5 s cutoff endpointing**; users report being talked over mid-thought; third-party "Say, Pi" extension exists just to add interruption. **Lesson: warmth cannot compensate for bad turn-taking — endpointing IS the product.** (maginative.com · App Store reviews)

### 12. Open-source / developer stacks (implementation-grade findings)

**Converged architecture across Pipecat, LiveKit, Deepgram Flux, Retell, ElevenLabs Agents:**
fast VAD (Silero, stop ~0.2–0.55 s) *triggers* a semantic end-of-turn classifier that either **commits** the turn or **extends the wait** up to a ceiling. Nobody ships a bare silence timer anymore.

- **Pipecat (Daily) — most borrowable.** Silero `VADParams(confidence=0.7, start_secs=0.2, stop_secs=0.2, min_volume=0.6)` + **smart-turn-v3**: Whisper-Tiny encoder + classifier, ~8M params, ONNX CPU (~10 ms inference), 8 s rolling window, 23 languages, `stop_secs=3.0` silence fallback. **BSD-2 code, weights, AND training data — directly reusable in openclaw-voice.** Interruption via `min_words` strategies; cancels LLM + flushes TTS on `StartInterruptionFrame`. Latency target: ~500–800 ms voice-to-voice (budget: STT 100 / LLM TTFT 80 / TTS TTFB 80 ms).
- **LiveKit Agents.** Silero defaults (`min_silence_duration=0.55`, `activation_threshold=0.5`) + turn-detector model gating endpointing between `min_delay=0.5` and `max_delay=3.0` s (0.3/2.5 with the audio detector) — plus a **dynamic mode** (EMA of the user's own pause statistics, alpha 0.9). **False-interruption recovery:** if barge-in speech dies out within 2 s and produced no words, the agent *resumes speaking* — the most polished recovery pattern surveyed. Client SDK exposes `initializing/idle/listening/thinking/speaking` as pushed state. ⚠️ Turn-detector **weights license-locked to LiveKit** — pattern borrowable, model not.
- **Deepgram Flux.** End-of-turn built into STT: `eot_threshold=0.7`, `eot_timeout_ms=5000`, plus **eager EOT** (speculative LLM start at lower confidence, ~150–250 ms saved, 50–70% more LLM calls) — the main trick for sub-500 ms feel. Emits `AgentThinking`/`AgentStartedSpeaking` + per-turn latency metrics over WS (validates our dev-overlay plan).
- **Vocode (MIT).** Transcriber-driven endpointing with punctuation-aware cutoffs; **backchannel filter**: interruptions ignored unless >3 words or non-backchannel regex (`m+-?hm+` etc.) — cheap, borrowable barge-in hygiene.
- **Retell.** `responsiveness` 0–1 + dynamic mode; `interruption_sensitivity`; `denoising_mode` incl. background-speech cancellation; backchannel emission ("yeah", "uh-huh") for perceived liveness.
- **ElevenLabs Agents.** `turn_timeout` default **7 s (range 1–30 s)** — the only product exposing a patience ceiling in our 15–20 s ballpark; `turn_eagerness: eager/normal/patient`; `interruption_ignore_terms`. Since we already pay for ElevenLabs TTS, their agents platform is the "buy" comparison — but it replaces our whole stack and violates local-STT privacy.

---

## Part 2 — Three candidate UI directions

*(All three assume the same engine work — M2 turn detection + barge-in — and differ in the M3 surface. All are dark-default, 4-state [IDLE/LISTENING/THINKING/SPEAKING], with a fixed kill control.)*

### Direction A — "Gauge" (full-screen state ring, driving-native)

One screen, one element: a large ring/orb centered, consuming ~60% of viewport, that IS the state display — dim amber breathing (idle/armed), **green ring that expands with live mic energy** (listening — motion tracks your voice, self-verifying à la Meta's banner but better), violet slow-orbit (thinking — calm ≤1 s, visibly spinning beyond, doubling as the latency indicator), blue waveform pulse (speaking). Bottom fifth of the screen is a **full-width red STOP bar** — the kill switch, thumb-reachable in a mount, no icon-hunting. Last user utterance + first line of the reply render as two large-type caption lines under the ring (Siri-captions pattern for "did it hear me right"), nothing else. No transcript, no settings on this surface (swipe up for the desktop layout).
**Pros:** maximum peripheral legibility (color+motion readable without focusing, NHTSA-friendly); zero cognitive load; kill switch is unmissable; simplest to build (one canvas element).
**Cons:** useless for kitchen/desktop reading; a second layout is mandatory from day one; risks feeling like a toy demo on desktop.

### Direction B — "Call Screen" (phone-call metaphor with Gemini's verb set)

Model the session as a call to Jarvis: header with session timer + connection dot, center state visual (smaller orb + live captions of BOTH sides scrolling call-style like Character.AI), and a bottom control row of three big round buttons: **HOLD** (mic muted, session alive — for drive-through windows, toll booths, passenger chats), **red END** (kill: stops audio + capture instantly), **PTT** (press-and-hold override for noisy moments — the Siri steering-wheel pattern, satisfying our desktop story too). Interruption stays voice-first; the buttons are fallbacks.
**Pros:** the mental model is pre-installed in every human ("I'm on a call with Jarvis"); Hold vs End distinction (Gemini's best idea) prevents the Perplexity forgotten-hot-mic foot-gun; PTT override = deterministic endpointing escape hatch when the car is loud.
**Cons:** three targets are smaller than one bar (fat-finger risk at 70 mph); call chrome wastes space on desktop; timer/dot invite glances that carry no action.

### Direction C — "Transcript-first" (voice as a layer, ChatGPT-2026 pattern)

The conversation transcript is the screen — large-type bubbles streaming in live (text chunks already stream in our protocol), with a **persistent state strip docked at the bottom**: colored edge-glow spanning the full width (listening = green glow breathing with mic energy, thinking = violet sweep, speaking = blue), a mute toggle, and a red square stop at the strip's right. Driving mode = the same layout with font size ×2 and transcript auto-collapsed to the last exchange.
**Pros:** one codebase, no second layout; best for kitchen (read a recipe step from 1 m) and desktop (dev overlay slots in naturally); transcript persistence = re-read/re-copy (the Humane lesson); closest to today's index.html — least build risk.
**Cons:** weakest driving surface — text implies reading; state strip at bottom edge is less peripherally legible than a 60%-viewport ring; kill switch smallest of the three directions.

### Recommendation: **A-core with B's verbs** ("Gauge + Hold/PTT"), C's layout as the non-driving mode

Build **Direction A as the driving surface**, but replace its single STOP bar with a split bottom bar: **red STOP taking ~70% width** + a **HOLD segment (~30%)** on the right (Gemini's insight that "pause without killing the session" is a real mid-drive need — killing the session to order a coffee, then re-arming by touch, violates our zero-touch goal). Long-press anywhere on the ring = PTT override (B's escape hatch, no extra chrome). Then implement **Direction C as the desktop/kitchen layout** — it's ~80% our existing index.html plus the state strip — with a layout toggle driven by viewport/URL param (`?mode=drive`).

**Why this pick.** (1) The post-mortems are unanimous that visible state — especially THINKING — is the trust-or-death feature (R1's missing thinking state, Pin's dead air); a 60%-viewport ring is the strongest possible state display and doubles as the latency indicator, so we get PRD §6's "latency absorbed peripherally" for free. (2) The kill switch requirement (PRD story 4) wants area, not icons — only A gives it a bar; B's three-button row is the fat-finger risk NHTSA glance discipline exists to prevent. (3) Hold and PTT are cheap to graft onto A (one split bar + one gesture) and buy us B's two genuinely load-bearing verbs while skipping its chrome. (4) C must exist anyway for kitchen/desktop (stories 11–12) and is the lowest-cost path from current code — making it the *only* surface, though, would ship our weakest driving UI for our primary use case. (5) Engine-wise the survey's converged architecture maps 1:1 onto our stack: keep Silero client/server, add **Pipecat smart-turn-v3** (BSD-2, CPU ONNX, ~10 ms) as the semantic gate, adopt LiveKit's min/max-delay endpointing envelope stretched to our 15–20 s patience ceiling, Vocode's backchannel regex filter + LiveKit's 2 s false-interruption resume for barge-in hygiene, and Deepgram-style per-turn latency events feeding the dev overlay. That is precisely M2 items 1–2 and 6, with every pattern license-clean.

**M3 build order implied:** state-event protocol additions (server pushes `state: listening|thinking|speaking` + per-turn latency) → Gauge canvas + kill/hold bar → C-layout retrofit of index.html → `?mode=` switch → parked-car night legibility test (PRD M3 exit).

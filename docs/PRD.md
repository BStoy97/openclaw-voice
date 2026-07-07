# OpenClaw Voice — Product Requirements Document (PRD)

**Version:** 1.0 — DRAFT (awaiting Nathan review)
**Last updated:** 2026-07-07
**Owner:** Nathan
**Status:** 🟡 Draft — Phases 1–2 of revival complete, Phase 5 (implementation) blocked on this PRD's approval
**Project type:** Self-host infra + personal daily-driver tool (upstream open-source adoption, not a SaaS product for now)
**Repo:** `~/projects/active/openclaw-voice/` · branch `dev/revive-2026-07` · upstream `Purple-Horizons/openclaw-voice`
**Companion docs:** `docs/CODEBASE-REVIEW.md` (current-state audit) · `docs/UI-RESEARCH.md` (UX patterns survey)

---

## 1. Product Summary

**What:** Browser-based, back-and-forth voice chat with Nathan's OpenClaw agent (Jarvis) — same memory, tools, and persona as text chat, spoken instead of typed. STT runs locally (faster-whisper on the Mac Studio; voice audio never leaves the machine), TTS via ElevenLabs streaming, LLM via the OpenClaw gateway's `/v1/chat/completions` endpoint.

**Who:** Nathan, primarily **while driving** (Central Florida commutes and errands — hands must stay on the wheel, eyes on the road). Secondarily: desktop at the office, walking around the house, kitchen while cooking.

**Why now:** Adopted Feb 2026 (chosen over Pipecat+Deepgram for privacy — local Whisper), then stalled. Nathan's daily workflow already runs through Jarvis via Signal text; voice is the missing modality, and driving is the one context where text is unusable and voice is strictly better. The codebase is a working demo away from a daily-driver tool: revival Phase 1 (move + rebuild) is done, the gap list is known (CODEBASE-REVIEW.md), and every gap is fixable in normal dev sessions.

**One-liner:** *Talk to Jarvis in the car like a passenger who happens to run your whole life.*

## 2. Primary Use Case: HANDS-FREE DRIVING

The design center. Every feature decision defers to "does this work at 70 mph on I-95 with the AC on?"

### 2.1 Interaction model
- **No push-to-talk. Ever.** The driving flow must never require a screen touch after the session starts. Session begins before the car moves (one tap or auto-start on page load) and runs until killed.
- **Full-duplex intent:** the app is always either listening or speaking, and can flip roles at any moment (see interruption, §2.7).

### 2.2 Turn detection & silence tolerance
- **Normal conversational pauses must not end Nathan's turn.** Target: the assistant tolerates **15–20 seconds of silence** mid-conversation before it assumes the turn is over and responds to what it has.
- This is *not* a fixed 20 s timeout on every utterance (that would add 20 s latency to "what time is it?"). Requirements:
  - **Semantic end-of-turn:** when the transcript-so-far is a complete question/command AND short silence follows (1.5–2.5 s), respond.
  - **Incomplete-utterance patience:** when the transcript ends mid-thought ("so what I'm thinking is…", trailing conjunctions, list-in-progress) or pitch/prosody signals continuation, keep listening up to the 15–20 s ceiling.
  - **VAD must be tunable** (silence ceiling, energy floor, semantic-mode on/off) via config and a dev overlay — not hardcoded constants (today: hardcoded 1.5 s amplitude gate; see CODEBASE-REVIEW §5).
  - "Still there?" prompt is acceptable at the ceiling; dead-air auto-response is not.

### 2.3 Car acoustics
- Assume: road noise (broadband, 60–75 dB cabin), wind buffeting, HVAC blower, occasional radio bleed, turn signals.
- Requirements:
  - **Adaptive noise floor:** energy-based gates must calibrate against a rolling noise-floor estimate, not an absolute threshold (today's `energy > 0.01` either never triggers or never releases in a car).
  - **Model-based VAD client-side or server-side in the loop** (Silero, not amplitude) as the speech/noise discriminator.
  - Browser constraints `echoCancellation`, `noiseSuppression`, `autoGainControl` enabled (partially present today).
  - Whisper model size selectable; expect `small`/`medium` (or `large-v3-turbo` if latency allows) needed for in-car WER targets, not `base`.
  - Radio/passenger speech is out of scope for v1 speaker isolation (see Non-goals); mitigation is "turn the radio down" + VAD tuning.

### 2.4 CarPlay / Android Auto
- **Explicit non-goal for v1–v3.** No native app; browsers don't run on CarPlay, and a native wrapper is a separate product decision.
- **The integration path that ships instead:** phone mounted or in pocket, audio routed to car speakers via **Bluetooth**, app running in Safari (iPhone) over Tailscale HTTPS. Must verify: WebAudio capture/playback continues with BT hands-free profile routing, and iOS Safari backgrounding behavior (see §2.5).
- Revisit CarPlay natively at M5+ only if the PWA path proves insufficient. Document findings either way.

### 2.5 Screen-off / phone-in-pocket mode
- Target: conversation continues with the **phone screen locked or in pocket**.
- Reality check (to validate in M2): iOS Safari suspends JS/WebAudio on lock. Mitigations in order of preference:
  1. **PWA installed to home screen** + audio session kept alive by continuous playback (silent keep-alive stream) — verify on iOS 19/26.
  2. Guided Access / screen-dim-but-on mode with the dark UI (screen technically on, ~zero glance value needed).
  3. If neither survives lock: document "screen dimmed, not locked" as the supported v1 mode and revisit native wrapper later.
- Wake Lock API (`navigator.wakeLock`) to prevent screen sleep during active sessions when screen-on mode is used.

### 2.6 Wake behavior
- **Auto-arm on load:** opening the page (re)connects and enters listening state after the first user gesture iOS requires (a single "Start" tap satisfies the mic-permission gesture; thereafter no touches).
- **Auto-resume:** on reconnect/refresh mid-drive, return to listening state without re-setup.
- **Wake word: non-goal for v1** (always-listening session model makes it redundant; wake-word engines in-browser are heavy). Reconsider for M5 ("hey Jarvis" via openWakeWord/Porcupine) if always-listening proves fatiguing or triggers too many false turns.

### 2.7 Interruption / barge-in (P0)
- Nathan speaking **over** TTS playback must, within ~300 ms: duck→stop playback, cancel remaining TTS synthesis and queued sentences, cancel the in-flight LLM stream if still generating, and begin capturing the new turn.
- Echo risk: with cabin speakers, TTS output re-enters the mic. Requires echo cancellation to be effective with BT routing, plus barge-in VAD threshold set above the residual-echo level; validate empirically in-car.
- A barge-in that was actually a cough/radio spike should degrade gracefully: if no transcribable speech follows within ~2 s, resume or offer "go on."

## 3. Secondary Use Cases

1. **Desktop (Mac, office):** highest audio quality, keyboard available. Wants: spacebar PTT *option* retained, transcript pane visible, latency overlay for dev work. This is also the dev/test environment.
2. **Walking around the house (phone):** short bursts, variable distance to phone, background TV/kids. Same continuous mode as driving, lower stakes; screen glanceable.
3. **Kitchen while cooking (phone/iPad on counter):** hands wet/dirty — no-touch is mandatory like driving, but the screen IS glanceable at arm's length: big-type transcript matters here most. Timers/unit conversions = frequent short turns; interruption used constantly.
4. (Deferred) Guests/family talking to household Jarvis — multi-user is explicitly out of scope for v1 (single shared conversation history today; see review §1).

## 4. User Stories (prioritized)

**P0 — driving alpha is unusable without these**
1. As a driver, I start a voice session with one tap before leaving the driveway, and never touch the screen again until I park.
2. As a driver, I pause mid-sentence for 10+ seconds to think (or merge onto a highway) and Jarvis waits instead of answering my half-formed thought.
3. As a driver, I talk over Jarvis mid-answer ("no — stop — I meant the OTHER meeting") and it stops talking within a beat and listens.
4. As a driver, I hit one huge always-visible button (or say nothing and flip the phone face-down… no — button, deterministic) and ALL audio stops instantly (kill switch).
5. As a driver, I know at a glance — peripheral vision only — whether Jarvis is listening, thinking, or talking (state must be legible as color/motion, not text).
6. As a driver on I-95 with AC on high, my speech still transcribes accurately enough that Jarvis doesn't answer a misheard question (in-car WER target §9).
7. As a user, when the connection drops in a dead zone, the app reconnects itself and tells me by voice ("back online") — no screen interaction.
8. As Nathan, my voice audio never leaves the Mac Studio (local STT is a hard requirement; only text goes to the LLM provider ElevenLabs gets only response text).

**P1 — daily-driver quality**
9. As a driver, Jarvis's answers are voice-length (a few sentences), not essays, and never read markdown junk aloud.
10. As a user, I ask a follow-up 30 seconds after the answer finished and the conversation context is intact.
11. As a kitchen user with wet hands, I can read the current answer as large text from 1 m away while it's being spoken.
12. As a desktop user, I can still hold spacebar to talk when I want precise turn control.
13. As Nathan, I can see per-turn STT/LLM/TTS latency in a dev overlay so we can tune the slow stage instead of guessing.
14. As a phone user, I reach the app over HTTPS from anywhere (Tailscale), because mobile browsers refuse mic access on plain HTTP.
15. As a user, if ElevenLabs is down or slow, I still get the answer spoken by a local fallback voice (degraded, not dead) — and the audio isn't garbled noise (int16/float32 bug, review §9).

**P2 — polish / later**
16. As a driver, the session auto-starts when my phone connects to the car's Bluetooth (shortcut/automation hook).
17. As Nathan, I can pick the voice (ElevenLabs voice ID) and Whisper model from a settings sheet instead of env vars.
18. As a user, I can say "new topic" to clear conversation context by voice.
19. As a household member, I can talk to Jarvis on the kitchen iPad under my own session, not Nathan's history (multi-session).
20. As Nathan, transcripts of voice sessions land in the same journal/memory pipeline as text chats.

## 5. Full Feature List

**Inherited from current codebase (keep, fix where noted in review):**
- Local Whisper STT (faster-whisper, model size configurable, `tiny`→`large-v3-turbo`)
- ElevenLabs streaming TTS (`eleven_turbo_v2_5`, sentence-by-sentence pipeline)
- Local TTS fallback ladder (Chatterbox/XTTS/mock) — fix the int16/float32 wire bug
- Silero VAD (server) — promote from advisory to load-bearing
- WebSocket protocol (`/ws`) with streaming text + audio events
- OpenClaw gateway backend (chatCompletions) with conversation history; direct-OpenAI fallback
- `clean_for_speech` markdown/URL/emoji stripping before TTS
- Continuous mode + push-to-talk mode; spacebar toggle (desktop)
- API-key auth scaffolding (tiers, rate limits, master key) — off by default locally
- Single-file browser client; React widget package (needs protocol update or deprecation)
- Docker/RunPod deploy path (currently broken; low priority — Mac Studio is the deploy target)

**New for driving mode (the M2–M3 build list):**
- Adaptive, tunable turn detection: rolling noise floor + model VAD + semantic end-of-turn + 15–20 s patience ceiling (§2.2–2.3)
- Barge-in: user speech cancels TTS + LLM stream and opens a new turn (§2.7)
- Wake-on-load auto-listen loop; auto-resume after reconnect, spoken reconnect notice (§2.6, story 7)
- Kill switch: single always-visible control that halts capture + playback + in-flight turn (§6)
- Driving UI: 4-state visual model (idle/listening/thinking/speaking) legible peripherally; dark default; big targets (§6)
- Latency instrumentation: per-turn STT/LLM/TTS/total, dev overlay + logged metrics (story 13)
- Mobile HTTPS via Tailscale Funnel (or Cloudflare tunnel — decide; §10 Q4), verified on iPhone Safari
- Screen-off/pocket mode per §2.5 (PWA + audio keep-alive, Wake Lock)
- Config surface: silence ceiling, VAD thresholds, Whisper model, voice ID — env + URL params + settings sheet (P2)
- Server fixes required underneath: `.env` loading, startup-crash fix, per-connection (or per-session) conversation state, WS error events + turn IDs (review §0, §4)

## 6. UI/UX Principles

- **Voice-first, glanceable second, touchable last.** Every interaction possible by voice; screen exists to reassure, not to operate.
- **One-glance state model:** four states — IDLE / LISTENING / THINKING / SPEAKING — mapped to distinct colors + motion (e.g., breathing ring vs. waveform vs. pulse), readable in peripheral vision at night without focusing. Text labels are secondary.
- **"Am I being listened to?" affordance:** the LISTENING state must be visually loud (large animated element reacting to live mic energy — motion correlates with your voice, which is self-verifying) and optionally audible (soft earcon on arm/disarm). Never a 12 px status string.
- **Big touch targets:** anything tappable while driving ≥ 88 pt; primary controls in the bottom third (thumb reach in a mount); zero interactions that require reading.
- **Kill switch:** one fixed-position, high-contrast (red), always-visible button — stops TTS playback, mic capture, and in-flight generation in one tap, session recoverable with one more tap. Also bound to spacebar/Esc on desktop. No confirmation dialog.
- **Dark mode default** (night driving; OLED). Light mode is the toggle, not the default. Avoid large white surfaces in any state.
- **Latency made visible peripherally:** THINKING state motion doubles as the latency indicator (calm ≤ 1 s, visibly "working" beyond); dev overlay shows numbers, driver UI never shows digits.
- **No walls of text while driving:** transcript pane collapses to last-exchange-only in driving layout; full history in desktop layout.
- **Fail loudly by voice, quietly on screen:** errors are spoken ("lost the server, retrying") and shown as state color, never as modal dialogs.

## 7. Non-Goals (v1–v3)

- ❌ **Native iOS/Android app, CarPlay/Android Auto app** — PWA-in-browser only (§2.4).
- ❌ **Wake word** — session model is always-listening; revisit M5 (§2.6).
- ❌ **Multi-user / multi-tenant hosting, billing tiers as a product** — auth scaffolding stays but productizing the hosted tier is out of scope for the revival (the in-repo pricing tables are upstream's ambitions, not ours).
- ❌ **Speaker diarization / radio-vs-Nathan isolation** — mitigate with VAD tuning + radio volume, not modeling.
- ❌ **Local LLM for the agent brain** — the backend is the OpenClaw gateway (Jarvis's normal model chain).
- ❌ **Replacing Signal text chat** — voice is an additional modality, not a migration.
- ❌ **Landing page / marketing work** — openclawvoice.com stays as-is (upstream's page) until Phase 5 is done; any change needs Nathan's approval as external-facing.
- ❌ **Telephony (Twilio dial-in)** — interesting later; not now.

## 8. Success Metrics

| Metric | Definition | Target (M2 alpha) | Target (M5) |
|---|---|---|---|
| Round-trip latency | end of user speech → first TTS audio audible, p50 (desktop LAN) | ≤ 2.0 s | ≤ 1.2 s |
| Round-trip latency, mobile | same, iPhone via Tailscale, p50 | ≤ 3.0 s | ≤ 2.0 s |
| Wake-to-first-word | page open → armed and listening | ≤ 5 s | ≤ 2 s |
| Barge-in response | user speech onset → TTS silenced | ≤ 500 ms | ≤ 300 ms |
| In-car transcription | WER on in-car recorded eval set (Nathan's voice, AC on, 45+ mph) | ≤ 15% | ≤ 8% |
| Turn-detection quality | % of turns cut off early OR answered during an intentional pause | ≤ 10% | ≤ 3% |
| Session robustness | 30-min drive with ≥1 coverage gap: session self-recovers without touch | 100% | 100% |
| Session length | typical usable session without restart/degradation | ≥ 30 min | ≥ 2 hr |
| Kill switch | tap → all audio stopped | ≤ 200 ms | ≤ 200 ms |
| Privacy invariant | voice audio bytes leaving Mac Studio | 0 | 0 |

Measurement: latency numbers come from the M2 instrumentation (story 13) — logged per turn, summarized per session; WER from a small recorded in-car eval set transcribed once per Whisper model candidate.

## 9. Open Questions (need Nathan)

1. **Fork or upstream?** Remote still points at `Purple-Horizons/openclaw-voice`. Our changes (driving mode) are opinionated — fork to `BStoy97/openclaw-voice` and optionally PR generic fixes upstream? (Recommended: fork; upstream looks dormant since Feb.)
2. **TTS spend ceiling.** ElevenLabs is paid/metered and driving mode will multiply usage (rough order: ~$0.10–0.20 per 10-min conversation on turbo pricing — verify at M2 with real metering). OK as-is, set a monthly cap, or prioritize a local-TTS-first mode (Piper/XTTS on the Studio) with ElevenLabs as the premium voice?
3. **Which phone mount reality?** Mounted-with-screen-visible vs. phone-in-pocket changes how hard we chase §2.5 screen-off mode in M2 vs. M4. What's the actual in-car setup?
4. **Tailscale Funnel vs. Cloudflare tunnel** for mobile HTTPS — Funnel is simplest and keeps traffic in the tailnet (recommended); Cloudflare matches existing 3apples.net infra. Preference? (Also: which port — 8765 is unregistered; needs a PORTS.md entry either way.)
5. **Whisper model + latency trade:** if `small`/`medium` is needed for in-car WER, STT time rises. Acceptable to spend up to ~1 s of the latency budget on STT, or should we buy accuracy with a GPU box later instead?
6. **Voice:** keep ElevenLabs "Jessica" (current hardcoded default) or pick a voice? (30-second decision, but it's hardcoded today and shouldn't be.)
7. **Python ceiling:** bump `requires-python` to `<3.15` and test on 3.14, or standardize on 3.12? (Old venv ran 3.14 out-of-range; we rebuilt on 3.12.)

## 10. Milestones

### M1 — Move + revive ✅ (done 2026-07-07, this revival's Phase 1–2)
- Repo at `~/projects/active/openclaw-voice/`, venv rebuilt (py3.12), baseline tests recorded (21P/3F/1S/5E), branch `dev/revive-2026-07`, codebase review shipped.
- Exit criteria met; remaining M1 leftovers roll into M2: fix the four §0 blockers from the review so the server actually boots from `.env`.

### M2 — Driving-mode alpha (core loop)
- Fix review §0 blockers (dotenv, double-prefix, startup crash, packaging) + commit the pending `backend.py` edit with tests
- Turn detection v2: adaptive noise floor, Silero in the loop, semantic end-of-turn, tunable 15–20 s patience (Phase 5 item 1)
- Barge-in (Phase 5 item 2) · wake-on-load auto-listen loop (item 3)
- Latency instrumentation + dev overlay (item 6)
- Test protocol per DEV-PROCESS.md; in-car eval recording made
- **Exit:** one real 20-min drive with zero screen touches, zero early cutoffs on deliberate 10 s pauses, working barge-in

### M3 — UI revamp
- Implement chosen direction from UI-RESEARCH.md: 4-state visual model, kill switch, dark default, driving layout vs. desktop layout
- React widget: update to streaming protocol or formally deprecate in favor of the vanilla client
- **Exit:** UI states legible in a parked-car night test; kill switch ≤ 200 ms

### M4 — Mobile HTTPS deploy
- Tailscale Funnel (per Q4 answer) serving the app to iPhone Safari; PWA manifest; Wake Lock; screen-off/pocket-mode findings documented (§2.5)
- launchd service on Mac Studio + PORTS.md registration
- **Exit:** full conversation on iPhone over cellular in the driveway; session survives a lock/unlock cycle (or documented limitation + chosen fallback mode)

### M5 — Public polish (optional / upstream-facing)
- Local-TTS-first mode (per Q2), voice/model settings sheet, wake-word spike, transcript→journal pipeline, upstream PRs for generic fixes, landing-page refresh (with approval)
- **Exit:** Nathan uses it weekly without babysitting; decision made on upstreaming vs. hard fork

---

*Phase 5 implementation order and test requirements: see `prompt.md` §PHASE 5 and `~/openclaw/docs/DEV-PROCESS.md`. Do not start M2 until Nathan reviews this PRD + CODEBASE-REVIEW.md + UI-RESEARCH.md.*

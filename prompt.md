# Task: Revive openclaw-voice, plan it properly, and modernize for hands-free driving use

You are Fable 5 working inside Nathan's OpenClaw workspace on his Mac Studio.
Nathan is the user. Follow ~/openclaw/AGENTS.md, ~/openclaw/SOUL.md, and ~/openclaw/USER.md
for style, safety, and workspace conventions. This is a multi-phase project — do
NOT ask permission to continue between phases; execute end-to-end, only pause for
questions when a decision genuinely blocks progress. Signal-formatting rules
apply: no markdown tables in chat replies; long deliverables go into files with
NavClaw URLs.

## PROJECT: openclaw-voice

Browser-based back-and-forth voice chat with Nathan's OpenClaw agent —
same memory, tools, persona as text chat, spoken instead of typed. Adopted
from the open-source upstream repo `Purple-Horizons/openclaw-voice` in
Feb 2026, then stalled. Reviving now.

## STATE ON DISK (verify each of these first, correct if drifted)

- Current checkout: `~/templates/openclaw-voice/`
- Git remote: `https://github.com/Purple-Horizons/openclaw-voice.git`
- Last commit locally: `28866f2 README: Comprehensive docs with all options`
- Local venv exists: `~/templates/openclaw-voice/.venv/` (python 3.14) — TRASH before move
- Populated `.env` exists (do NOT overwrite; migrate as-is)
- Landing page domain: openclawvoice.com
- Target new home: `~/projects/active/openclaw-voice/` (this file lives there)

### Source tree
- `src/server/main.py` — WebSocket entry
- `src/server/stt.py`, `tts.py`, `vad.py`, `streaming.py`, `backend.py`,
  `auth.py`, `text_utils.py`
- `src/client/index.html` — browser UI
- `packages/react/` — react component package
- `scripts/generate_master_key.py`, `scripts/download_models.py`
- `tests/test_auth.py`, `test_server.py`, `test_modules.py`
- `deploy/runpod/README.md` — RunPod deploy notes
- `pyproject.toml`, `requirements.txt`, `uv.lock`
- `Dockerfile`, `docker-compose.yml`

### Existing docs (only these — no PRD exists)
- `~/templates/openclaw-voice/README.md` — feature list, config table,
  arch diagram, WebSocket API, roadmap
- `~/templates/openclaw-voice/SKILL.md` — setup/how-to for OpenClaw
- `~/templates/openclaw-voice/docs/twitter-article.md` — marketing copy
- `~/templates/openclaw-voice/docs/index.html` + `CNAME` — landing page
- `~/templates/openclaw-voice/deploy/runpod/README.md` — deploy guide

### Historical context in the workspace
- `~/life/journals/2026-02-28.md` — day repo was found, group "J: Voice
  Feature" created, decision to pick openclaw-voice over Pipecat+Deepgram
  (privacy: local Whisper)
- `~/openclaw/memory/archive/2026-02-28.md` — same, expanded
- `~/openclaw/memory/archive/2026-03-01.md` — status: repo cloned to
  `~/openclaw/voice-chat/` (that path is gone; active checkout moved to
  `~/templates/openclaw-voice/`), chatCompletions endpoint enabled in
  gateway, ElevenLabs credentials at `~/.openclaw/credentials/elevenlabs.json`
- `~/life/areas/server-operations.md` — brief server-ops references
- No entry in `~/openclaw/backlog/`, `~/openclaw/reports/`, or
  `~/life/projects/`. No PRD anywhere.

## STACK (from README)

- STT: faster-whisper (local; voice never leaves Mac Studio)
- TTS: ElevenLabs streaming (`eleven_turbo_v2_5`) — key at
  `~/.openclaw/credentials/elevenlabs.json`; approval required per use
  (see MEMORY.md — ⚠️ ElevenLabs is paid/metered)
- VAD: Silero
- Transport: WebSocket at `/ws`
- Backend: OpenClaw gateway `http://localhost:18789` via
  `/v1/chat/completions` — endpoint already enabled

## PHASE 1 — Move the project (no code changes yet)

1. **Trash the stale venv first:**
   `rm -rf ~/templates/openclaw-voice/.venv` — its shebangs and
   editable-install paths would break on move anyway. Clean start at
   the destination.
2. `~/projects/active/openclaw-voice/` already exists (this prompt file
   lives inside it). Move the remaining working tree contents from
   `~/templates/openclaw-voice/` into it — preserve `.git/`, `.env`,
   and every source/doc file. If `mv` complains about the non-empty
   destination, move files individually and keep `prompt.md` in place.
3. If any `launchd` plist, cron job, or script references the old path,
   grep for `templates/openclaw-voice` across:
   - `~/openclaw/`
   - `~/.openclaw/`
   - `~/Library/LaunchAgents/`
   - `~/bin/`
   - `~/life/`
   and update each to the new path. Report each change.
4. Git remote — confirm it still points to
   `https://github.com/Purple-Horizons/openclaw-voice.git`. If the remote
   should be re-pointed to a Nathan-owned fork, ASK before creating one;
   otherwise leave upstream as-is.
5. Rebuild the venv cleanly at the destination:
   `cd ~/projects/active/openclaw-voice && python3 -m venv .venv &&
   source .venv/bin/activate && pip install -e . && pip install -r
   requirements.txt`. Reuse Python 3.14 if still available; otherwise
   fall back to the highest 3.10–3.13 that satisfies `pyproject.toml`
   (`requires-python = ">=3.10,<3.14"`). Note: the old venv used 3.14
   which is OUTSIDE the declared range — flag this as a `pyproject.toml`
   bump candidate.
6. Run `pytest tests/` — record pass/fail baseline before any changes.
7. Commit the move + venv rebuild on a new branch `dev/revive-2026-07`.
   Do NOT push. Follow the workspace's git deployment safety standard
   (main is live by default; work on dev/*).

## PHASE 2 — Codebase review

Produce a review doc at
`~/projects/active/openclaw-voice/docs/CODEBASE-REVIEW.md`:

- Architecture summary (module-by-module, ~1 paragraph each for the 8
  files in `src/server/` and the react package)
- Dependency freshness: check `pyproject.toml` + `requirements.txt`
  against latest releases; flag any >12mo stale or with known CVEs
- Test coverage: what's covered, what isn't. Actual coverage % if you
  can run `coverage.py` without pain.
- WebSocket protocol audit: is it robust against dropped/duplicate
  messages? Does the client reconnect?
- VAD behavior: current end-of-utterance timeout and how it's tuned
- Security: `auth.py` — is the master-key flow safe? `.env` handling?
- Deploy story: Dockerfile + docker-compose + runpod — do they still
  work? Note issues without fixing yet.
- Landing page (`docs/index.html`, CNAME) — is it live at
  openclawvoice.com right now?
- Known bugs / rough edges you spotted

Under 800 lines; bullet-heavy.

## PHASE 3 — PRD

Write `~/projects/active/openclaw-voice/docs/PRD.md`. Model the format
after the workspace's existing PRDs, especially:
- `~/projects/active/dockside/docs/PRD.md`
- `~/projects/active/tradecomms/docs/PRD.md`
- `~/projects/active/clawcam/docs/PRD.md`

Read at least one of those before drafting so the structure matches.

Required PRD sections:

1. **Product summary** — what it is, who uses it, why now
2. **Primary use case: HANDS-FREE DRIVING**
   - No push-to-talk
   - Silence tolerance in normal conversation flow: 15–20 seconds is
     acceptable before the assistant assumes the turn is over. Design
     the VAD / turn-detection to respect that.
   - Car acoustics: road noise, wind, HVAC, radio bleed
   - CarPlay / Android Auto integration path (or explicit non-goal)
   - Screen-off mode for iPhone-in-pocket use
   - Wake behavior: does it auto-start on page load? Wake word?
   - Interruption handling: user can talk over the assistant to
     cancel/redirect
3. **Secondary use cases** — desktop use, walking around the house,
   kitchen while cooking
4. **User stories** — at least 12, prioritized P0/P1/P2
5. **Full feature list** — every capability the app should have,
   inheriting from current README + roadmap, PLUS everything new for
   driving mode
6. **UI/UX principles**
   - Voice-first, glanceable UI
   - Big touch targets (driving safety)
   - Dark mode default (night driving)
   - Latency indicators the driver can absorb peripherally
   - "Am I being listened to?" affordance without staring at screen
   - Explicit "Kill switch" that stops all audio in one tap
7. **Non-goals** — call out what this is NOT
8. **Success metrics** — round-trip latency target, transcription WER
   in-car, session length, wake-to-first-word time
9. **Open questions** — flag anything requiring Nathan's decision
10. **Milestones** — M1 (move + revive), M2 (driving-mode alpha), M3
    (UI revamp), M4 (mobile HTTPS deploy), M5 (public polish)

## PHASE 4 — UI research

Before touching UI code, research how modern voice-chat apps handle
hands-free/in-car UX. Save findings to
`~/projects/active/openclaw-voice/docs/UI-RESEARCH.md`.

Look at (or their public documentation of):

- ChatGPT Voice Mode (Advanced Voice)
- Grok voice
- Google Gemini Live
- Meta AI voice
- Perplexity Voice
- Siri hands-free, Google Assistant driving mode
- Rabbit R1, Humane Pin (post-mortem lessons)
- Character.AI voice
- Pi (Inflection)
- Open-source references: Vocode, Pipecat, LiveKit Voice, Retell,
  Deepgram Voice Agent, ElevenLabs Conversational AI

For each, capture:
- Turn-detection strategy (VAD, wake word, silence timeout)
- Interruption handling
- Visual state model (states + transitions)
- Kill-switch / mute affordance
- Latency budget they publish (or you can infer)
- Anything screenshot-worthy — grab screenshots to
  `docs/ui-research/screenshots/` where possible

Deliverable: 3 candidate UI directions for openclaw-voice with a
recommended pick and rationale (~1500 words + screenshots).

## PHASE 5 — Implementation

Only proceed after Nathan reviews PHASE 2–4 output. Do NOT skip his
review. When resuming, work in these order:

1. **Turn detection for driving mode** — VAD tunable up to 20s
   silence tolerance; end-of-turn detection needs pitch/prosody hints,
   not just amplitude
2. **Barge-in / interruption** — user speech mid-TTS cancels TTS
   playback and captures new turn
3. **Wake-on-load + auto-listen loop** — no button required after
   initial permission grant
4. **New UI** based on chosen direction from PHASE 4
5. **Kill switch** (large red target, always visible)
6. **Latency instrumentation** — expose STT/LLM/TTS times in dev overlay
7. **Mobile HTTPS access** — Tailscale Funnel is the simplest; verify
   works on iPhone

Every implementation change requires tests per
`~/openclaw/docs/DEV-PROCESS.md`. Run `pytest tests/` after each
sub-step; auto-fix loop up to 10 attempts, then escalate.

## CONSTRAINTS AND SAFETY

- Do NOT change `~/openclaw/openclaw.json` config without showing Nathan
  a diff and waiting for approval (per AGENTS.md config change rule)
- ElevenLabs API is paid — do not run TTS smoke tests without Nathan's
  explicit "do it!" (per MEMORY.md rule)
- Do NOT push to `origin/main`. Work on `dev/revive-2026-07` throughout
- Do NOT modify installed packages under `.venv/` (anti-delusion rule)
- Do NOT publish landing-page changes to openclawvoice.com without
  approval — that's a public-facing external change
- All new/moved files need NavClaw URLs when reported back to Nathan:
  format `https://files-jarvis.3apples.net/edit/openclaw/<path>`
  (only works for paths under `~/openclaw/`; adapt for
  `~/projects/active/openclaw-voice/` — use
  `https://files-jarvis.3apples.net/edit/projects/active/openclaw-voice/<path>`
  if NavClaw serves that root, otherwise report the absolute path and
  note NavClaw coverage gap)
- If NavClaw doesn't yet serve `~/projects/active/openclaw-voice/`, ADD
  it (config path is in NavClaw's config file — search for it) and
  redeploy NavClaw as a small side-task; commit that separately
- Update `~/openclaw/MEMORY.md` after PHASE 3 (add a Voice section
  pointing to the new PRD and repo location)
- Create `~/life/projects/openclaw-voice.md` as a proper project file
  with goals, status, decisions, next steps (per PARA structure in
  AGENTS.md)
- Add a backlog entry in `~/openclaw/backlog/active.md`

## OUTPUT FORMAT AT END

When done with phases 1–4, send Nathan a single Signal-formatted
summary (≤ 29 chars/line, no tables, stacked cards / `━━━` dividers)
with:
- One-line status of each phase
- NavClaw or absolute URL for each deliverable
- Top 3 open questions blocking PHASE 5
- Proposed next action

Do NOT ask "should I keep going?" or offer to stop — per SOUL.md,
propose the next concrete action instead.

Now begin.

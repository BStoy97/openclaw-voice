# openclaw-voice — Codebase Review

**Date:** 2026-07-07 · **Branch:** `dev/revive-2026-07` · **Baseline commit:** `28866f2` (+ `ff015bc` move commit)
**Reviewer:** Jarvis (revival Phase 2)
**Verdict up front:** solid skeleton, real streaming pipeline, but the server **does not currently start** with the documented `.env`-only setup (see Blockers), the deploy story is broken, and turn-detection is nowhere near driving-mode requirements. All fixable.

---

## 0. Revival Blockers (fix these first in Phase 5)

1. **`.env` is never loaded.** `python-dotenv` is in requirements but `load_dotenv()` is never called anywhere in `src/`. Only pydantic-settings reads `.env`, and only for its own fields.
2. **Double-prefix bug.** `Settings` has `env_prefix = "OPENCLAW_"`, so the field `openclaw_gateway_url` maps to env var `OPENCLAW_OPENCLAW_GATEWAY_URL`. The documented `OPENCLAW_GATEWAY_URL` in `.env` never reaches settings; the `os.getenv("OPENCLAW_GATEWAY_URL")` fallback in `main.py:110` only sees *process* env, which `.env` never populates (see #1). Same story for `ELEVENLABS_API_KEY` (read via `os.environ` in `tts.py:34`).
3. **Startup crash without credentials.** With no gateway vars and no `OPENAI_API_KEY` in process env, `AIBackend._setup_client()` calls `AsyncOpenAI(api_key=None)` which **raises `OpenAIError` at construction** in current openai SDK versions → FastAPI lifespan startup fails → server never binds. Verified live on this machine 2026-07-07 (`uvicorn` startup traceback ends in `openai/_client.py: raise OpenAIError`). The "echo fallback" in `backend.chat()` is unreachable because the exception isn't caught.
   - Net effect: README quick-start (`cp .env.example .env` → run) produces a server that crashes on boot. This is also why 5 of the 29 tests error (server fixture times out).
4. **`pip install -e .` is broken** (pre-existing): hatchling can't infer a wheel target because there is no `openclaw_voice/` package dir (code is `src/server/`, `src/client/`). Needs `[tool.hatch.build.targets.wheel]` config. Tests dodge this via `sys.path.insert`, but the **Dockerfile and CI use `-e .` and therefore fail**.

---

## 1. Architecture summary (module by module)

### `src/server/main.py` (388 lines)
FastAPI app + the whole conversation loop in one WebSocket handler. Startup initializes four globals (STT, TTS, backend, VAD). The `/ws` handler drives a strict half-duplex state machine: client sends `start_listening` → base64 float32 PCM `audio` chunks accumulate in a Python list → `stop_listening` triggers transcribe → streamed LLM chat → inline sentence-boundary splitting → per-sentence TTS → base64 `audio_chunk` messages back. Also hosts a small key-management REST API (`POST /api/keys`, `GET /api/usage`) and serves the demo page. Issues: uses deprecated `@app.on_event("startup")`; serves `FileResponse("src/client/index.html")` relative to CWD (breaks when launched from another directory); sentence-splitting logic is duplicated here rather than using `streaming.py`; the whole turn is synchronous within the receive loop, so nothing can interrupt TTS (no barge-in).

### `src/server/stt.py` (109 lines)
`WhisperSTT` with graceful backend ladder: faster-whisper → openai-whisper → mock. Auto device selection (CUDA → CPU int8 on Mac; MPS deliberately mapped to CPU for faster-whisper/CTranslate2). Transcription runs in a thread-pool executor (`run_in_executor`), `beam_size=5`, built-in `vad_filter=True`. Language hardcoded `en`. Clean module; the mock fallback keeps tests hermetic. Verified live: loads faster-whisper `tiny` in ~1.5s on the Mac Studio.

### `src/server/tts.py` (169 lines)
Misleadingly named `ChatterboxTTS` — actually a backend ladder: ElevenLabs (if `ELEVENLABS_API_KEY` in process env) → Chatterbox → Coqui XTTS → mock silence. ElevenLabs path uses `eleven_turbo_v2_5`, `pcm_24000`, with a true streaming generator. Red flags: **runtime `subprocess pip install elevenlabs`** if the SDK is missing (never do this in prod, and it violates our anti-delusion rule about mutating installed envs); the ElevenLabs SDK calls are **sync** generators driven inside async code (blocks the event loop between chunks); local backends return float32 arrays whose bytes get sent where the browser client decodes **int16** — i.e., fallback TTS audio is decoded as garbage noise by the streaming client path (only ElevenLabs pcm_16 output matches the client decoder). Sample rate is hardcoded 24000 in messages regardless of backend.

### `src/server/vad.py` (45 lines)
Thin Silero VAD wrapper via `torch.hub` (downloads from GitHub at first run — network dependency at startup). Single method `is_speech(chunk)`, threshold 0.5, fails **open** (returns True on any error, VAD absent). Note: recent Silero versions enforce fixed 512-sample windows @16kHz; the server feeds it whatever chunk size the browser sent (4096 samples), so on current silero this likely throws per-chunk and permanently returns True — needs runtime verification. Crucially: **server VAD is advisory only** — its result is just echoed to the client as `vad_status`; it plays no role in turn-taking.

### `src/server/backend.py` (167 lines)
`AIBackend` wraps AsyncOpenAI for both direct OpenAI and the OpenClaw gateway (OpenAI-compatible). Keeps `conversation_history` in the instance, truncated to last 10 messages per request. **The instance is a process-global shared by every WebSocket client** — all connected users share one conversation history (privacy/correctness bug the moment there's a second client; also unbounded memory growth since history is never truncated in storage, only per-request). Has both `chat()` and `chat_stream()`. Error handling swallows API errors into a spoken apology (good for UX, hides root cause — should log + surface a structured error event too).
**Uncommitted local edit (pre-existing, reviewed):** implements the previously-stub `openclaw` backend type with a real AsyncOpenAI client and loosens `chat`/`chat_stream` gating from `backend_type == "openai"` to `self._client` truthiness. The edit is sound and should be committed in Phase 5 (with a test). Note its `base_url=f"{self.url}/v1"` while the "openai"-type gateway path in `main.py` already appends `/v1` — pick one convention.

### `src/server/streaming.py` (182 lines)
**Dead code.** Sentence-splitter, `stream_openai_response`, `StreamingTTS`, `process_with_streaming` — none of it is imported by `main.py`, which reimplements the same logic inline (with a different splitting algorithm). Two implementations of the same concern will drift; delete or consolidate in Phase 5. Its `max_tokens=150` also disagrees with backend.py's 500.

### `src/server/auth.py` (243 lines)
Telegram-bot-style token system: `ocv_`-prefixed keys, SHA-256 hashes stored (not plaintext — good), per-minute rate limiting, monthly minute quotas, tiers (free/pro/enterprise) with pricing table. All **in-memory** — every key except the env master key evaporates on restart, so `/api/keys` is a demo, not a product. Rate-limit window resets are per-key naive timestamps (fine at this scale). `record_usage`/`check_monthly_quota` exist but are **never called** from the WebSocket path — minutes are never actually metered.

### `src/server/text_utils.py` (90 lines)
`clean_for_speech()`: strips code blocks, markdown, hashtags, URLs, a hardcoded emoji list; converts bullets to "Next, ". Reasonable single-pass regex pipeline. Zero test coverage (see §3). Edge cases: nested/unbalanced markdown, the "Next," suffix trim only handles one trailing occurrence, emoji stripping is a fixed set rather than a Unicode-category sweep.

### `src/client/index.html` (739 lines)
Single-file UI: push-to-talk button, continuous-mode toggle, transcript pane, streaming text + queued audio playback. Uses deprecated `ScriptProcessorNode` (4096-sample buffers) — should be AudioWorklet. Client-side turn detection in continuous mode is a **hardcoded amplitude-energy gate** (`energy > 0.01`, `silenceThreshold = 1500 ms`) — see §5. Reconnects on close every 2s (flat interval, forever). Renders assistant markdown with a hand-rolled regex renderer into `innerHTML` — XSS risk if the model emits crafted HTML: user text is escaped, but assistant `renderMarkdown()` does **not** escape HTML before injecting. Audio decode path assumes int16 PCM for `audio_chunk` and float32 for legacy `audio_response` — inconsistent by design, breaks non-ElevenLabs backends (see tts.py note).

### `packages/react/` (`@openclaw/voice-widget-react`, 256 lines)
Self-contained `<VoiceWidget>` with the same mic capture approach (also `ScriptProcessorNode`). **Protocol drift:** it only handles the legacy `response_text` / `audio_response` messages — it ignores `response_chunk`, `audio_chunk`, `response_complete`, so against the current server it shows nothing and plays nothing during streaming responses. Also `btoa(String.fromCharCode(...bytes))` spreads a 16KB array per frame (stack-risk pattern), and there's no `dist/` — the package has never been built/published.

---

## 2. Dependency freshness

Fresh venv (2026-07-07, Python 3.12.12) resolves all floors to current releases — the *installed* set is fresh. The problem is the **declared floors** (Jan-2024 era, ~18 months stale), which both under-protect (admit CVE-affected versions) and over-admit (major-version jumps the code was never written for):

- `openai>=1.6.0` → installs **2.44.0** (major bump). Constructor now raises without a key — this is Blocker #3. Code was written against 1.x idioms; needs a compatibility pass + pin `>=2,<3`.
- `elevenlabs>=1.0.0` → installs **2.56.0** (major bump). `text_to_speech.convert(...)` still exists but the 2.x SDK reshuffled several call signatures — needs a live smoke test (paid — Nathan approval) + pin.
- `transformers>=4.36.0` → installs **5.13.0** (major bump); only used transitively for optional local TTS. Consider making it an extra.
- `torch>=2.1.0` floor admits versions vulnerable to **CVE-2025-32434** (`torch.load` RCE, CVSS 9.3, fixed in 2.6.0 — verified via GitHub advisory GHSA-53q9-r3pm-6pq6 / NVD). Installed 2.12.1 is fine; raise the floor to `>=2.6`.
- `python-multipart>=0.0.6` floor admits **CVE-2024-24762** (Content-Type ReDoS, fixed 0.0.7, regression-fixed 0.0.8 — verified via NVD/GitHub advisory). Raise floor to `>=0.0.8` (installed is current).
- `numpy>=1.26.0` → installs **2.4.6** (numpy 2.x ABI). `np.frombuffer`/`concatenate` usage is fine, but faster-whisper/torch interplay on numpy 2 should be smoke-tested.
- `fastapi>=0.109.0` → **0.139.0**: `on_event` deprecation (warning now, removal later) — migrate to lifespan handler.
- `websockets>=12.0` → **16.0**: only used by tests; API used still present.
- Duplication: deps are declared in **both** `pyproject.toml` and `requirements.txt` with drift (requirements adds librosa/soundfile/webrtcvad/silero-vad/pyyaml/transformers/whisper that pyproject lacks; `webrtcvad` and `librosa` appear entirely unused in `src/`). Consolidate to pyproject + `uv.lock` (a `uv.lock` already exists but predates the move — regenerate).
- `requires-python = ">=3.10,<3.14"` while the old venv ran 3.14: either bump the ceiling after a test pass on 3.13/3.14, or keep 3.12 and delete the confusion. Flagged as pyproject bump candidate.

## 3. Test coverage

- 29 collected: **21 pass, 3 fail, 1 skip, 5 error** (baseline, fresh venv 2026-07-07).
  - 3 failures = `TestAIBackend` — the openai 2.x constructor crash (Blocker #3), not logic regressions.
  - 5 errors = `test_server.py` fixture: spawned uvicorn dies on the same startup crash → "Server did not start in time". The 15s boot wait would also be too short for a first-run whisper model download even after the fix.
  - 1 skip = conditional (port in use / backend-dependent).
- `coverage.py` (src/, files actually imported by tests):
  - `auth.py` 87% · `vad.py` 83% · `stt.py` 46% · `tts.py` 36% · `backend.py` 35% · overall **54%**
  - `main.py`, `streaming.py`, `text_utils.py`: **0% — never imported by any unit test.** True whole-tree coverage is roughly a third.
- What's genuinely covered: token manager lifecycle (generate/validate/revoke/rate-limit/quota), VAD init + speech/silence classification, STT/TTS init ladders + mock paths.
- Not covered at all: the WebSocket protocol handler (the actual product), sentence-splitting, `clean_for_speech`, auth *enforcement* on `/ws` and `/api/keys`, reconnect behavior, react widget (no JS tests exist).
- CI (`.github/workflows/test.yml`): 3.10/3.11/3.12 matrix using `uv pip install -e ".[dev,stt]"` — **fails on Blocker #4** (editable install). CI has presumably been red since inception.

## 4. WebSocket protocol audit

- Transport: JSON text frames both directions; audio as base64 float32 up, base64 int16 down. ~33% base64 overhead on a hot path — fine for LAN, wasteful for cellular; binary frames or WebRTC (roadmap item) would halve bandwidth.
- **Dropped messages:** no sequence numbers, no acks, no resume. A disconnect mid-response loses the rest of the response silently (client reconnects to a blank-slate connection; server-side per-connection state is just discarded — though conversation history survives in the shared global backend, see §1 backend).
- **Duplicate messages:** `start_listening` twice just resets the buffer (harmless); duplicate `stop_listening` triggers a second empty-buffer pass (harmless); duplicate `audio` frames are concatenated — no idempotency, but consequences are mild given half-duplex design.
- **Ordering:** relies entirely on TCP/WS ordering; audio chunks have no timestamps or indices. Client playback queue assumes in-order arrival (true for a single WS).
- **Backpressure:** none. Server `send_json`s every TTS chunk as fast as ElevenLabs yields; per-chunk `vad_status` messages spam the socket once per 4096-sample frame (~4/sec per client) with no client consumer benefit.
- **Errors:** generic `except Exception` → log + close, no error frame to the client; client can't distinguish crash from network loss.
- **Reconnect:** client auto-reconnects every 2s flat, forever (no backoff, no jitter, no max). Auth-failure closes (4001/4002/4003) correctly do NOT reconnect. React widget does **not** auto-reconnect at all.
- **Keepalive:** client pings every 30s; server answers pong. No server-side idle timeout.
- Verdict: fine for a demo on localhost; needs sequence/turn IDs, an error event type, resumable turns, and barge-in signaling for the driving use case.

## 5. VAD / turn-detection behavior (key gap for driving mode)

- **The real end-of-utterance detector is client-side, not Silero.** In continuous mode, `index.html` computes mean-abs energy per 4096-sample frame; `energy > 0.01` = voice; **1500 ms** below threshold = turn over (`silenceThreshold = 1500`, hardcoded, not configurable via UI/env/URL param).
- Push-to-talk mode: turn boundary = button release, VAD irrelevant.
- Server Silero VAD: per-chunk classification echoed back as `vad_status`; the client ignores it. Threshold 0.5 default, not configurable. Possible chunk-size incompatibility with current silero (see §1 vad.py).
- Consequences for the car: a fixed amplitude gate will (a) never detect silence over road/HVAC noise → recording never auto-stops, or (b) with cabin quiet + thoughtful 5-second pauses, cut Nathan off 10× too early. The PRD's 15–20 s tolerance requires semantic/prosodic end-of-turn detection (or at minimum: adaptive noise-floor calibration + much longer, configurable timeout + Silero on the client via ONNX/WASM).
- Auto-listen loop exists (continuous mode re-arms 300 ms after playback ends) but there's **no barge-in**: while TTS plays, the mic is not captured, so you cannot talk over the assistant to cancel it. Interruption = Phase 5 item #2.

## 6. Security review

- **Auth default-off** (`require_auth: false`), and the demo/dev posture leaks into prod paths:
  - `/api/keys` with auth off: anyone who can reach the port can mint keys (harmless-ish while keys gate nothing, but it's a trap once auth flips on).
  - Master key comparison is `!=` on strings (non-constant-time; low practical risk, easy fix with `secrets.compare_digest`).
  - `main.py`'s docstring advertises `-H "x-master-key: ..."` but the code reads a **query param** — docs/code mismatch, and secrets in query strings end up in access logs and browser history. Same for WS `?api_key=` (header fallback exists; prefer it).
- **Key storage:** SHA-256 of keys only (good); keys are high-entropy (`secrets.token_urlsafe(32)`, good); in-memory only (restart wipes tenants — MVP-acceptable, flag for prod).
- **Master-key flow:** `generate_master_key.py` produces `ocv_master_<urlsafe32>` — sound. But `load_keys_from_env` registers the master key as an enterprise *client* key too, so the admin credential doubles as a usage credential (scope-separation smell).
- **`.env` handling:** populated `.env` migrated intact; correctly gitignored; `.env.example` committed. Ironically the code never loads it (Blocker #1) — the current "leak surface" is zero because nothing reads it. After the dotenv fix, add a startup log that never echoes values.
- **CSWSH / origin:** no `Origin` checking on the WebSocket. With auth off, any website Nathan visits could connect to `localhost:8765/ws` from his browser and use the mic-less API (send text-less audio, read responses). Real risk for a long-running local service; check Origin or require auth even locally.
- **XSS:** assistant markdown rendered to `innerHTML` without HTML-escaping first (client, §1). Model output is semi-trusted at best; escape then render.
- **TTS runtime `pip install`:** a server that installs packages at runtime is a supply-chain and reproducibility hazard. Remove.
- **CORS:** not configured (FastAPI default = no CORS middleware) — fine since same-origin serving is the model.

## 7. Deploy story (Docker / compose / RunPod)

- **Dockerfile: broken, multiple independent ways.**
  - Base image `nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04` — NVIDIA's tag scheme uses full patch versions (`12.1.1-…`), and old CUDA tags get pruned from Docker Hub; this tag almost certainly no longer pulls. Needs re-verification and likely a bump to a current CUDA 12.x tag.
  - `uv pip install -e ".[stt]"` — dies on Blocker #4 (hatchling packaging).
  - `ENV PATH="/root/.cargo/bin:$PATH"` — current uv installer places the binary in `~/.local/bin`; the build would not find `uv` even if the image pulled.
  - Copies `src/` but not `scripts/`; `CMD` is correct for the layout.
  - Irrelevant to the Mac Studio host anyway (CUDA image; no NVIDIA runtime on macOS) — Docker path is for cloud GPU deploys only.
- **docker-compose.yml:** `version: '3.8'` is obsolete (warning); GPU service + CPU profile split is sensible; healthchecks fine; inherits all Dockerfile breakage.
- **RunPod guide:** references `ghcr.io/purple-horizons/openclaw-voice:latest` — no evidence a published image exists (no publish workflow in `.github/`); pricing/GPU table is Feb-2026 era; instructions are otherwise coherent. Treat as aspirational docs.
- **Local (the path that matters for Nathan):** run under the venv with env vars exported; needs a launchd plist + PORTS.md registration when it becomes a service (port 8765 currently unregistered — check PORTS.md before binding; 3400-3499 range may be more appropriate per convention).
- CI: red (see §3).

## 8. Landing page

- **Live right now:** `https://openclawvoice.com` → HTTP 200 via GitHub Pages, `last-modified: Sun, 01 Feb 2026`, served from `docs/` (CNAME file present, correct). Full SEO/OG/Twitter meta, `og-image.png` present.
- Served from the **upstream Purple-Horizons repo's** Pages config (domain + content controlled by wherever GitHub Pages is configured — presumably upstream repo settings, not anything on this machine). Any landing-page change requires push access + Nathan approval (external-facing).
- `docs/twitter-article.md` is marketing copy, unused by the page.

## 9. Known bugs / rough edges (consolidated)

- **P0 / blocks revival**
  - `.env` never loaded + double-prefix bug → documented setup can't configure anything (Blockers 1–2)
  - Startup crash with no key (Blocker 3) — also breaks 8 tests and any "clone and run" user
  - `pip install -e .` / Docker / CI all broken on hatchling packaging (Blocker 4)
- **P1 / blocks driving mode**
  - Turn detection: hardcoded 1.5 s amplitude gate, unusable in cars (§5)
  - No barge-in / interruption (§5)
  - Non-ElevenLabs TTS audio decoded as int16 garbage by streaming client (float32/int16 mismatch)
  - React widget ignores streaming protocol entirely (dead against current server)
  - Shared global conversation history across all clients (backend.py)
- **P2 / quality**
  - `streaming.py` dead code duplicating main.py logic (with drift: max_tokens 150 vs 500)
  - `vad_status` spam ~4 msg/sec; Silero chunk-size mismatch suspicion
  - Runtime `pip install` in tts.py; sync ElevenLabs SDK blocking the event loop
  - `ScriptProcessorNode` deprecated in both clients; innerHTML XSS in markdown renderer
  - `FileResponse` CWD-relative path; `on_event` deprecation; usage metering never invoked; master key in query params; no WS Origin check; no reconnect backoff; react widget never reconnects
  - Dependency floors admit CVE-2025-32434 (torch <2.6) and CVE-2024-24762 (python-multipart <0.0.7) — installed versions fine, floors need raising
  - `requirements.txt` vs `pyproject.toml` drift; `webrtcvad` + `librosa` unused; stale `uv.lock`
  - `requires-python <3.14` vs the 3.14 venv it used to run under (bump candidate)
  - Language hardcoded `en`; voice ID hardcoded (Jessica); no config for VAD threshold/silence timeout

## 10. What's genuinely good (keep)

- Clean module boundaries; every external dependency has a mock fallback → tests run anywhere
- Sentence-level streaming TTS pipeline design is right (first-token-to-first-audio is the latency win)
- Local Whisper STT (privacy: voice never leaves the Mac) — the original reason this repo was chosen over Pipecat+Deepgram
- Key hashing, tiered auth scaffolding, and the WS close-code convention (4001/4002/4003) are sensible foundations
- README/SKILL docs are honest and match the code's intent, if not always its behavior

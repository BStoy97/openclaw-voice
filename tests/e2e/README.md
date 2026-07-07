# E2E — Chrome-driven end-to-end test

Manual/agent-driven E2E protocol, first run 2026-07-07 (all scenarios PASS).

## Setup
1. Start the server keyless (echo backend, deterministic):
   `env -u OPENAI_API_KEY -u OPENCLAW_GATEWAY_URL -u OPENCLAW_GATEWAY_TOKEN -u ELEVENLABS_API_KEY OPENCLAW_STT_MODEL=tiny .venv/bin/python -m uvicorn src.server.main:app --port 8801`
2. Copy `fixtures/*.wav` into `src/client/` so the page can fetch them same-origin (remove after).
3. Open `http://127.0.0.1:8801/?mode=desk` in Chrome.
4. Inject the fake microphone (dev console or automation): override
   `navigator.mediaDevices.getUserMedia` to return a `MediaStreamDestination`
   stream from a 16 kHz `AudioContext`; `__speak(name)` plays a fixture WAV
   buffer into it. Do NOT `await ctx.resume()` inside the override (it hangs
   without user activation — resume contexts from a real click instead).
5. Click START (must be a real/trusted click for audio-context activation).

## Fixtures (Piper en_US-amy-medium, generated via scripts in repo)
- `question.wav` — "What time is it right now?" (complete utterance)
- `long-question.wav` — "Tell me a very long story about a lobster who lives in Florida."
- `incomplete.wav` — "So what I was thinking was" (mid-thought, tests semantic patience)

## Scenarios + 2026-07-07 results
1. **Full loop**: speak `question` → states listening→thinking→speaking→listening;
   transcript verbatim; reply audible. PASS (whisper-tiny transcribed exactly).
2. **Barge-in**: speak `long-question`, then `question` during SPEAKING →
   playback silenced, new turn processed. PASS (left speaking 428 ms after
   barge onset; both turns answered).
3. **Kill switch**: STOP during SPEAKING → all audio halted. PASS (26 ms).
4. **Latency overlay** (press L): turn_metrics render. PASS
   (stt 135 ms · llm_ttft 0 ms (echo) · tts first chunk 740 ms · total 901 ms).
5. **Semantic patience**: `incomplete` fixture stays PENDING ~9 s before
   commit vs ~2 s for complete utterances; all commits reason=semantic,
   ceiling never hit. PASS.
6. **Variants**: ?variant=a (ring) / b (edge-glow + status word) / c (orb)
   all render with identical STOP/HOLD controls. PASS.

## Known limitation
Real-microphone acoustics (echo cancellation vs cabin speakers, BT routing,
iOS Safari) are NOT covered — needs the M4 phone-in-car pass.

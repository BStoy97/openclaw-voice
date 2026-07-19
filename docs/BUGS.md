# openclaw-voice Bug Log

## BUG-001 — Mic dead after iOS audio interruption resume

**Date:** 2026-07-14  
**Session:** 856998ec-aa18-4eaa-8f7a-cb5275a308c9  
**Severity:** High (session becomes one-way, user must kill and restart)

### Reproduction

1. Start an active voice session on iOS
2. Have another audio app playing in the background
3. iOS switches audio routing to the background app (~8:05 AM incident — could be a phone call, music, etc.)
4. Voice session fully pauses
5. After ~1 min, switch back to the openclaw-voice app
6. **Result:** TTS resumes from where it paused ✅ but mic input is dead ❌

### Observed behavior

- TTS playback resumed mid-sentence from the buffer — that part worked correctly
- Microphone input was silently broken — app could not hear user at all
- No error surfaced in the UI

### Hypothesis

iOS fires an audio session interruption notification (`AVAudioSession.interruptionNotification`) when another app takes the audio focus. On resume, the app needs to:

1. Call `audioSession.setActive(true)` to re-acquire the session
2. Restart the input stream / WebSocket audio ingestion

The current keep-alive mechanism re-pins the **output** channel (TTS/Bluetooth), but likely doesn't re-acquire **input** after an interruption ends. The mic stream just stays dead silently.

### Fix direction

Handle `AVAudioSessionInterruptionTypeEnded` on the client side:
- Re-activate the audio session
- Restart the mic capture / send a re-subscribe signal to the server
- Possibly trigger a brief "I'm back, can you hear me?" prompt to confirm two-way comms restored

### Related

- Keep-alive stream work (nightwork 2026-07-13) — fixed output channel pinning; input not covered
- Client session resume registry (10-min grace window) — handles reconnect identity, not audio routing

### RESOLUTION (2026-07-15)

**Status: FIXED** — shipped in `8b70078`/`c939d03` build set. Root cause confirmed
from field logs: iOS kills the `getUserMedia` capture track on audio-route theft
(this is a web app — the AVAudioSession hypothesis above maps to the browser's
MediaStreamTrack layer). Fix was three-layered, client-side in `index.html`:

1. `track.onended` → immediate mic rebuild
2. Watchdog: worklet frames stopping >2.5s while capturing → full teardown + re-acquire
3. `startMic()` stale-ready trap fixed (STOP/START now verifies `track.readyState === 'live'`)

Verified in the 2026-07-16 drive: zero mic deaths across the session (the same
route-theft scenario did not recur; watchdog armed). The "I'm back" confirmation
prompt idea was not implemented — the dev-flash shows "mic rebuilt" instead.

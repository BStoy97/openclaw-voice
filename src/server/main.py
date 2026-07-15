"""
OpenClaw Voice Server

WebSocket server that handles:
- Audio input from browser
- Speech-to-Text via Whisper
- AI backend communication
- Text-to-Speech via ElevenLabs
- Audio streaming back to browser
"""

import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from pydantic_settings import BaseSettings

from .stt import WhisperSTT
from .tts import ChatterboxTTS
from .backend import AIBackend
from .vad import VoiceActivityDetector
from .turn import (
    BARGE_CANDIDATE,
    BARGE_IN,
    EOT_PENDING,
    TURN_COMMITTED,
    USER_SPEECH_STARTED,
    TurnConfig,
    TurnEngine,
)
from .auth import token_manager, load_keys_from_env, APIKey
from .text_utils import clean_for_speech

load_dotenv()

# --- Field diagnostics ------------------------------------------------------
# Structured JSONL event log for in-car test review (transcripts, barge
# decisions, timings, lifecycle). ON by default while driving-mode testing
# is active; disable with OPENCLAW_VOICE_DEBUG=0 once operations are stable.
_VOICE_DEBUG = os.getenv("OPENCLAW_VOICE_DEBUG", "1").strip().lower() not in ("0", "false", "off")
_VOICE_LOG_PATH = os.path.expanduser(
    os.getenv("OPENCLAW_VOICE_LOG", "~/Library/Logs/openclaw-voice-events.jsonl")
)


def voice_log(event: str, **fields) -> None:
    """Append one structured diagnostic event. Never raises."""
    if not _VOICE_DEBUG:
        return
    try:
        import datetime as _dt
        rec = {"ts": _dt.datetime.now().isoformat(timespec="milliseconds"), "event": event}
        rec.update(fields)
        with open(_VOICE_LOG_PATH, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


class Settings(BaseSettings):
    """Server configuration."""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8765
    
    # Auth
    require_auth: bool = False  # Set True for production
    master_key: Optional[str] = None  # Admin key for full access
    
    # STT
    stt_model: str = "base"  # tiny, base, small, medium, large-v3-turbo
    stt_device: str = "auto"  # auto, cpu, cuda, mps
    
    # TTS
    tts_model: str = "chatterbox"
    tts_voice: Optional[str] = None  # Path to voice sample for cloning
    
    # AI Backend
    backend_type: str = "openai"  # openai, openclaw, custom
    backend_url: str = "https://api.openai.com/v1"
    backend_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    
    # OpenClaw Gateway (auto-detected from OPENCLAW_GATEWAY_URL + TOKEN)
    gateway_url: Optional[str] = None
    gateway_token: Optional[str] = None
    
    # Audio
    sample_rate: int = 16000
    
    class Config:
        env_prefix = "OPENCLAW_"
        env_file = ".env"


settings = Settings()
app = FastAPI(title="OpenClaw Voice", version="0.1.0")

# Global instances (initialized on startup)
stt: Optional[WhisperSTT] = None
barge_stt: Optional[WhisperSTT] = None
tts: Optional[ChatterboxTTS] = None
backend: Optional[AIBackend] = None
vad: Optional[VoiceActivityDetector] = None


@app.on_event("startup")
async def startup():
    """Initialize models on server start."""
    global stt, barge_stt, tts, backend, vad
    
    logger.info("Initializing OpenClaw Voice server...")
    
    # Load API keys
    load_keys_from_env()
    if settings.require_auth:
        logger.info("🔐 Authentication ENABLED")
    else:
        logger.warning("⚠️ Authentication DISABLED (dev mode)")
    
    # Initialize STT
    logger.info(f"Loading STT model: {settings.stt_model}")
    stt = WhisperSTT(
        model_name=settings.stt_model,
        device=settings.stt_device,
    )
    
    # Barge-candidate STT: reuses the main model by default. A separate
    # smaller model sounds attractive latency-wise, but two CTranslate2
    # models in one process deadlock in native code (observed 2026-07-15:
    # whole event loop wedged mid-response). Opt in explicitly via
    # OPENCLAW_BARGE_STT_MODEL only for supervised experiments.
    barge_model = os.getenv("OPENCLAW_BARGE_STT_MODEL", "")
    if barge_model and barge_model != settings.stt_model:
        logger.warning(f"Loading separate barge STT model: {barge_model} (experimental)")
        barge_stt = WhisperSTT(model_name=barge_model, device=settings.stt_device)
    else:
        barge_stt = stt

    # Initialize TTS
    logger.info(f"Loading TTS model: {settings.tts_model}")
    tts = ChatterboxTTS(
        voice_sample=settings.tts_voice,
    )
    
    # Initialize AI backend
    # Auto-detect OpenClaw gateway
    gateway_url = settings.gateway_url or os.getenv("OPENCLAW_GATEWAY_URL")
    gateway_token = settings.gateway_token or os.getenv("OPENCLAW_GATEWAY_TOKEN")
    
    if gateway_url and gateway_token:
        # Use OpenClaw gateway (connects to Aria!)
        logger.info(f"🦞 Connecting to OpenClaw gateway: {gateway_url}")
        backend = AIBackend(
            backend_type="openai",  # Gateway speaks OpenAI API
            url=f"{gateway_url}/v1",
            model="openclaw:voice",  # Maps to 'voice' agent in config
            api_key=gateway_token,
            system_prompt=(
                "This conversation is happening via real-time voice chat. "
                "Keep responses concise and conversational — a few sentences "
                "at most unless the topic genuinely needs depth. "
                "No markdown, bullet points, code blocks, or special formatting."
            ),
        )
    else:
        # Fallback to direct OpenAI
        logger.info(f"Connecting to backend: {settings.backend_type}")
        backend = AIBackend(
            backend_type=settings.backend_type,
            url=settings.backend_url,
            model=settings.backend_model,
            api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY"),
        )
    
    # Warm up STT: the first CTranslate2 call JIT-initializes kernels and
    # was costing the first real turn ~2-4s extra ("slow to get going").
    try:
        t0 = time.monotonic()
        await stt.transcribe(np.zeros(8000, dtype=np.float32))
        logger.info(f"STT warm-up done in {time.monotonic() - t0:.1f}s")
    except Exception as e:
        logger.warning(f"STT warm-up failed (non-fatal): {e}")

    # Initialize VAD
    logger.info("Loading VAD model")
    vad = VoiceActivityDetector()
    
    logger.info("✅ OpenClaw Voice server ready!")


@app.get("/")
@app.get("/voice")
@app.get("/voice/")
async def index():
    """Serve the demo page."""
    return FileResponse(str(Path(__file__).parent.parent / "client" / "index.html"))


@app.post("/api/keys")
async def create_api_key(
    name: str,
    tier: str = "free",
    master_key: Optional[str] = None,
):
    """
    Create a new API key (requires master key).
    
    curl -X POST "http://localhost:8765/api/keys?name=myapp&tier=pro" \
         -H "x-master-key: YOUR_MASTER_KEY"
    """
    # Verify master key
    if settings.require_auth:
        if not master_key and not settings.master_key:
            return {"error": "Master key required"}
        
        provided_key = master_key or ""
        if provided_key != settings.master_key:
            # Also check if it's a valid master-tier key
            key = token_manager.validate_key(provided_key)
            if not key or key.tier != "enterprise":
                return {"error": "Invalid master key"}
    
    from .auth import PRICING_TIERS
    
    if tier not in PRICING_TIERS:
        return {"error": f"Invalid tier. Options: {list(PRICING_TIERS.keys())}"}
    
    tier_config = PRICING_TIERS[tier]
    
    plaintext_key, api_key = token_manager.generate_key(
        name=name,
        tier=tier,
        rate_limit=tier_config["rate_limit"],
        monthly_minutes=tier_config["monthly_minutes"],
    )
    
    return {
        "api_key": plaintext_key,  # Only shown once!
        "key_id": api_key.key_id,
        "name": api_key.name,
        "tier": api_key.tier,
        "monthly_minutes": api_key.monthly_minutes,
        "rate_limit": api_key.rate_limit_per_minute,
    }


@app.get("/api/usage")
async def get_usage(api_key: str):
    """
    Get usage stats for an API key.
    
    curl "http://localhost:8765/api/usage?api_key=ocv_xxx"
    """
    key = token_manager.validate_key(api_key)
    if not key:
        return {"error": "Invalid API key"}
    
    return token_manager.get_usage(key)


def _decode_audio(msg: dict) -> np.ndarray:
    """Decode a base64 float32 PCM `audio` message payload."""
    audio_bytes = base64.b64decode(msg["data"])
    return np.frombuffer(audio_bytes, dtype=np.float32)


async def run_turn(
    websocket: WebSocket,
    audio_data: np.ndarray,
    conn_backend: Optional[AIBackend],
    *,
    turn_id: Optional[int] = None,
    continuous: bool = False,
    t_commit: Optional[float] = None,
    stop_phrases=None,
    silence_secs: Optional[float] = None,
) -> float:
    """One full response turn: STT → streamed LLM → sentence-level TTS.

    Used by both the legacy stop_listening path and continuous-mode commits.
    Cancellation-safe: a barge-in cancels the surrounding task at any await.
    Returns the seconds of TTS audio sent (0.0 if none) so the caller can
    bound its wait for client playback.
    """
    t_commit = t_commit if t_commit is not None else time.monotonic()
    sample_rate = getattr(tts, "sample_rate", 24000) if tts else 24000

    def _tag(payload: dict) -> dict:
        if turn_id is not None:
            payload["turn_id"] = turn_id
        return payload

    logger.debug("Transcribing audio...")
    transcript = await stt.transcribe(audio_data)
    t_transcript = time.monotonic()

    await websocket.send_json(_tag({
        "type": "transcript",
        "text": transcript,
        "final": True,
    }))
    logger.info(f"Transcript: {transcript}")
    voice_log("transcript", turn_id=turn_id, text=transcript[:200],
              stt_ms=round((t_transcript - t_commit) * 1000))

    if not transcript.strip():
        return 0.0

    # A committed turn that is nothing but a stop phrase ("stop", "hold on")
    # is leftover interrupt intent — usually speech that landed just after
    # the barge window closed. Never forward it to the LLM as a prompt.
    if (
        continuous
        and stop_phrases
        and len(transcript.split()) <= 4
        and decide_barge(transcript, stop_phrases) == "cancel"
    ):
        logger.info(f"Bare stop phrase committed as turn; discarding: '{transcript.strip()}'")
        await websocket.send_json(_tag({"type": "tts_cancelled"}))
        return 0.0

    if continuous:
        await websocket.send_json(_tag({"type": "state", "state": "thinking"}))

    full_response = ""
    sentence_buffer = ""
    t_llm_first: Optional[float] = None
    t_first_audio: Optional[float] = None
    audio_bytes_sent = 0

    async def speak(text: str) -> None:
        """Synthesize one sentence and stream it as audio_chunk messages."""
        nonlocal t_first_audio, audio_bytes_sent
        speech_text = clean_for_speech(text)
        if not speech_text:
            return
        logger.debug(f"Synthesizing: {speech_text[:50]}...")
        async for audio_chunk in tts.synthesize_stream(speech_text):
            if t_first_audio is None:
                t_first_audio = time.monotonic()
                if continuous:
                    await websocket.send_json(
                        _tag({"type": "state", "state": "speaking"})
                    )
            audio_bytes_sent += len(audio_chunk)
            await websocket.send_json(_tag({
                "type": "audio_chunk",
                "data": base64.b64encode(audio_chunk).decode(),
                "sample_rate": sample_rate,
            }))

    # Stream response and synthesize sentences as they complete
    logger.debug("Streaming AI response...")
    async for chunk in conn_backend.chat_stream(transcript):
        if t_llm_first is None:
            t_llm_first = time.monotonic()
        full_response += chunk
        sentence_buffer += chunk

        # Send text chunk for progressive display
        await websocket.send_json(_tag({
            "type": "response_chunk",
            "text": chunk,
        }))

        # Check for sentence boundaries
        seps = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        while any(sep in sentence_buffer for sep in seps):
            # Find first sentence boundary
            earliest_idx = len(sentence_buffer)
            for sep in seps:
                idx = sentence_buffer.find(sep)
                if idx != -1 and idx < earliest_idx:
                    earliest_idx = idx + len(sep)

            if earliest_idx < len(sentence_buffer):
                sentence = sentence_buffer[:earliest_idx].strip()
                sentence_buffer = sentence_buffer[earliest_idx:]
                if sentence:
                    await speak(sentence)
            else:
                break

    # Handle any remaining text
    if sentence_buffer.strip():
        await speak(sentence_buffer.strip())

    # Signal end of response
    await websocket.send_json(_tag({
        "type": "response_complete",
        "text": full_response,
    }))
    logger.info(f"Response complete: {full_response[:100]}...")
    voice_log("response_complete", turn_id=turn_id, chars=len(full_response),
              audio_secs=round(audio_bytes_sent / 2 / sample_rate, 2))

    if continuous:
        t_done = time.monotonic()
        stt_ms = round((t_transcript - t_commit) * 1000)
        llm_ttft_ms = (
            round((t_llm_first - t_transcript) * 1000)
            if t_llm_first is not None else None
        )
        tts_first_ms = (
            round((t_first_audio - (t_llm_first or t_transcript)) * 1000)
            if t_first_audio is not None else None
        )
        commit_to_audio_ms = (
            round((t_first_audio - t_commit) * 1000)
            if t_first_audio is not None else None
        )
        # Perceived gap = user's silence before the engine committed
        # + commit-to-first-audio. THE number for "how long until it talks".
        perceived_ms = (
            round(silence_secs * 1000) + commit_to_audio_ms
            if (silence_secs is not None and commit_to_audio_ms is not None)
            else None
        )
        voice_log("turn_metrics", turn_id=turn_id, stt_ms=stt_ms,
                  llm_ttft_ms=llm_ttft_ms, tts_first_chunk_ms=tts_first_ms,
                  silence_before_commit_ms=(round(silence_secs * 1000) if silence_secs is not None else None),
                  commit_to_audio_ms=commit_to_audio_ms,
                  perceived_gap_ms=perceived_ms)
        await websocket.send_json(_tag({
            "type": "turn_metrics",
            "stt_ms": stt_ms,
            "llm_ttft_ms": llm_ttft_ms,
            "tts_first_chunk_ms": tts_first_ms,
            "perceived_gap_ms": perceived_ms,
            "total_ms": round((t_done - t_commit) * 1000),
        }))

    return audio_bytes_sent / 2 / sample_rate  # int16 → seconds of audio sent


class _SessionState:
    """Per-WebSocket-connection state."""

    def __init__(self):
        self.mode = "ptt"  # "ptt" (legacy) or "continuous" (server-driven)
        self.engine: Optional[TurnEngine] = None
        self.conn_backend: Optional[AIBackend] = None
        self.response_task: Optional[asyncio.Task] = None
        self.turn_counter = 0
        self.current_turn_id: Optional[int] = None
        self.client_id: Optional[str] = None
        self.playback_done: Optional[asyncio.Event] = None
        self.barge_decision_task: Optional[asyncio.Task] = None


# --- Reconnect session continuity -------------------------------------------
#
# Field data: mobile WebSocket drops (iOS screen-lock suspend, radio dead
# zones) used to blow away conversation_history on every reconnect — one
# 30-min phone test produced 29 gateway sessions. Clients that pass a
# persistent client_id in session_start can now resume the parked
# _SessionState (and its conn_backend conversation history) within a grace
# window instead of starting a brand-new conversation.

SESSION_REGISTRY_CAP = 32
SESSION_GRACE_SECS_DEFAULT = 600.0
SESSION_GRACE_SECS_MAX = 3600.0

# client_id -> {"session": _SessionState, "expires_at": float (monotonic)}
_SESSION_REGISTRY: dict = {}


def _session_grace_secs() -> float:
    """Grace window (seconds) a disconnected session stays resumable.

    Read fresh from the environment on every call (not cached at import)
    so it can be tuned without a process restart, and so unit tests can
    exercise it via monkeypatched env vars.
    """
    raw = os.getenv("OPENCLAW_SESSION_GRACE_SECS")
    try:
        val = float(raw) if raw is not None else SESSION_GRACE_SECS_DEFAULT
    except (TypeError, ValueError):
        val = SESSION_GRACE_SECS_DEFAULT
    return max(0.0, min(val, SESSION_GRACE_SECS_MAX))


def _purge_expired_registry(now: Optional[float] = None) -> None:
    """Lazy sweep: drop registry entries past their expiry.

    Called on every session_start (and before parking a new entry) so the
    registry never needs a background task.
    """
    now = time.monotonic() if now is None else now
    expired = [
        cid for cid, entry in _SESSION_REGISTRY.items()
        if entry["expires_at"] <= now
    ]
    for cid in expired:
        del _SESSION_REGISTRY[cid]


def _cap_session_registry() -> None:
    """Enforce the max registry size, dropping the oldest-parked entries
    first. Plain dict insertion order tracks park order here since entries
    are only ever inserted (never reordered) by `_park_session`."""
    while len(_SESSION_REGISTRY) > SESSION_REGISTRY_CAP:
        oldest_id = next(iter(_SESSION_REGISTRY))
        del _SESSION_REGISTRY[oldest_id]


def _park_session(session: "_SessionState") -> None:
    """Park a disconnected session's state in the resume registry.

    No-op if the connection never adopted a client_id (legacy / non-resuming
    clients get fresh state on every reconnect, as before).
    """
    client_id = session.client_id
    if not client_id:
        return
    _purge_expired_registry()
    grace = _session_grace_secs()
    _SESSION_REGISTRY[client_id] = {
        "session": session,
        "expires_at": time.monotonic() + grace,
    }
    _cap_session_registry()
    logger.info(f"Parked session for resume: client_id={client_id} grace={grace}s")


# --- Transcribe-then-decide barge-in -----------------------------------------
#
# Field data (truck testing): while the agent speaks over Bluetooth, iOS echo
# cancellation ducks the user's speech so hard that pure-VAD barge-in almost
# never fires. The TurnEngine now emits BARGE_CANDIDATE (buffered audio) after
# a short low-threshold speech run; we transcribe it and decide:
#
#   stop phrase heard  -> 'cancel'   kill playback, discard the audio
#   >= 2 words         -> 'takeover' kill playback, the speech becomes the
#                                    user's next turn (buffer seeds it)
#   empty / 1-2 fillers -> 'ignore'  agent keeps talking
#
# The engine's own 1.2 s pure-VAD BARGE_IN remains as a fallback when
# transcription is unavailable.

_DEFAULT_STOP_PHRASES = TurnConfig().stop_phrase_list


def _normalize_speech(text: str) -> str:
    """Lowercase, strip punctuation (apostrophes kept), collapse whitespace."""
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


_BARGE_FILLER_WORDS = frozenset(
    "um uh uhh er hmm hm mm mhm ah oh okay ok yeah well like so right sure".split()
)


def decide_barge(
    text: str, stop_phrases: Optional[Sequence[str]] = None
) -> str:
    """Classify a barge-candidate transcript: 'cancel' | 'takeover' | 'ignore'.

    Stop phrases match word-boundary-aware ("stop" never matches
    "stopwatch") anywhere in the utterance. Anything else with >= 2 words
    of which at least one is a content (non-filler) word is a genuine
    talk-over; filler-only fragments and single words are noise.
    """
    normalized = _normalize_speech(text or "")
    if not normalized:
        return "ignore"
    phrases = _DEFAULT_STOP_PHRASES if stop_phrases is None else stop_phrases
    for phrase in phrases:
        p = _normalize_speech(phrase)
        if p and re.search(r"\b" + re.escape(p) + r"\b", normalized):
            return "cancel"
    words = normalized.split()
    content = [w for w in words if w not in _BARGE_FILLER_WORDS]
    if len(words) >= 2 and len(content) >= 1:
        return "takeover"
    return "ignore"


async def _cancel_response_task(session: _SessionState) -> bool:
    """Cancel the in-flight response pipeline, if any. Returns True if one
    was actually cancelled.

    Bounded wait: TTS-subprocess generator cleanup can stall on
    cancellation (observed with the manual interrupt mid-generation) —
    never let that block the interrupt path. The task's finally-cleanup
    still runs whenever it does finish dying.
    """
    task = session.response_task
    if task is None or task.done():
        return False
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass
    except Exception as e:  # noqa: BLE001 - task errors already logged there
        logger.debug(f"Cancelled response task raised: {e}")
    return True


def _launch_response(
    websocket: WebSocket, session: _SessionState, audio: np.ndarray, t_commit: float,
    silence_secs: Optional[float] = None,
) -> None:
    """Start the response pipeline as a cancellable background task."""
    turn_id = session.current_turn_id
    engine = session.engine
    engine.set_agent_responding(True)
    session.playback_done = asyncio.Event()

    async def _task():
        try:
            sent_secs = await run_turn(
                websocket,
                audio,
                session.conn_backend,
                turn_id=turn_id,
                continuous=True,
                t_commit=t_commit,
                stop_phrases=engine.config.stop_phrase_list,
                silence_secs=silence_secs,
            )
            # TTS synthesizes faster than realtime, so sending finishes long
            # before the client's speakers do. Keep the barge window open
            # until the client reports playback_done (or a bounded estimate
            # elapses) so "stop" works for the WHOLE audible reply.
            if sent_secs and sent_secs > 0:
                try:
                    await asyncio.wait_for(
                        session.playback_done.wait(),
                        timeout=max(3.0, sent_secs + 3.0),
                    )
                except asyncio.TimeoutError:
                    pass
            await websocket.send_json({"type": "state", "state": "listening"})
        except asyncio.CancelledError:
            raise
        except WebSocketDisconnect:
            pass
        except Exception as e:  # noqa: BLE001 - keep the connection alive
            logger.error(f"Response pipeline error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "turn_id": turn_id,
                    "message": "response pipeline failed",
                })
                await websocket.send_json({"type": "state", "state": "listening"})
            except Exception:
                pass
        finally:
            engine.set_agent_responding(False)

    session.response_task = asyncio.create_task(_task())


async def _handle_barge_candidate(
    websocket: WebSocket, session: _SessionState, event
) -> None:
    """Transcribe a BARGE_CANDIDATE's buffered audio and resolve it.

    Deliberately does NOT resolve the candidate when transcription is
    unavailable or fails — leaving it unresolved lets the engine's 1.2 s
    pure-VAD BARGE_IN fallback fire.
    """
    engine = session.engine
    if engine is None:
        return
    candidate_stt = barge_stt or stt
    if candidate_stt is None or event.audio is None or len(event.audio) == 0:
        return  # unresolved -> pure-VAD fallback handles it
    t_cand = time.monotonic()
    try:
        text = await asyncio.wait_for(
            candidate_stt.transcribe(event.audio), timeout=2.5
        )
    except asyncio.TimeoutError:
        voice_log("barge_transcribe_timeout", buffer_secs=round(len(event.audio) / 16000, 2))
        logger.warning("Barge-candidate transcription timed out (2.5s); relying on pure-VAD fallback")
        return
    except Exception as e:  # noqa: BLE001 - degrade to the fallback path
        voice_log("barge_transcribe_error", error=str(e)[:120])
        logger.debug(f"Barge-candidate transcription failed: {e}")
        return
    voice_log("barge_decision",
              text=text.strip()[:80],
              decision=decide_barge(text, engine.config.stop_phrase_list),
              transcribe_ms=round((time.monotonic() - t_cand) * 1000))
    decision = decide_barge(text, engine.config.stop_phrase_list)
    now = time.monotonic()

    if decision == "cancel":
        # Stop phrase: kill playback, discard the audio (not a new turn).
        # Always send tts_cancelled — even when the send pipeline already
        # finished, the client still has queued audio to flush.
        await _cancel_response_task(session)
        await websocket.send_json({"type": "tts_cancelled"})
        await websocket.send_json({"type": "state", "state": "listening"})
        engine.resolve_barge("cancel", now)
        engine.set_agent_responding(False)
        logger.info(f"Stop-phrase barge: '{text.strip()}'")
        voice_log("stop_phrase_cancel", text=text.strip()[:80])
    elif decision == "takeover":
        # Genuine talk-over: the buffered speech becomes the new user turn.
        await _cancel_response_task(session)
        await websocket.send_json({"type": "tts_cancelled"})
        engine.resolve_barge("takeover", now)
        engine.set_agent_responding(False)
        session.turn_counter += 1
        session.current_turn_id = session.turn_counter
        await websocket.send_json({
            "type": "turn_started",
            "turn_id": session.current_turn_id,
        })
        logger.info(f"Takeover barge: '{text.strip()}'")
        voice_log("takeover_barge", text=text.strip()[:80])
    else:
        engine.resolve_barge("ignore", now)
        logger.debug(f"Ignored barge candidate: '{text.strip()}'")


async def _handle_turn_events(
    websocket: WebSocket, session: _SessionState, events
) -> None:
    """Translate TurnEngine events into protocol messages / actions."""
    for event in events:
        if event.type == USER_SPEECH_STARTED:
            session.turn_counter += 1
            session.current_turn_id = session.turn_counter
            await websocket.send_json({
                "type": "turn_started",
                "turn_id": session.current_turn_id,
            })
        elif event.type == EOT_PENDING:
            await websocket.send_json({
                "type": "eot_pending",
                "turn_id": session.current_turn_id,
            })
        elif event.type == BARGE_CANDIDATE:
            # Fire-and-forget with an in-flight guard: transcription must
            # never block the receive loop (a hung whisper call wedged the
            # whole pipeline in field testing 2026-07-15), and must never
            # pile up. If it times out, the engine's 1.2 s pure-VAD
            # fallback still interrupts on sustained speech.
            if session.barge_decision_task is None or session.barge_decision_task.done():
                session.barge_decision_task = asyncio.create_task(
                    _handle_barge_candidate(websocket, session, event)
                )
            else:
                voice_log("barge_candidate_skipped", reason="decision in flight")
        elif event.type == BARGE_IN:
            # Legacy pure-VAD fallback (candidates went unresolved).
            logger.info("Barge-in: cancelling agent response")
            voice_log("vad_fallback_barge_in")
            if await _cancel_response_task(session):
                await websocket.send_json({"type": "tts_cancelled"})
        elif event.type == TURN_COMMITTED:
            t_commit = time.monotonic()
            await websocket.send_json({
                "type": "turn_committed",
                "reason": event.reason,
                "turn_id": session.current_turn_id,
            })
            logger.info(
                f"Turn {session.current_turn_id} committed "
                f"(reason={event.reason}, {len(event.audio)} samples)"
            )
            if event.audio is not None and len(event.audio) > 0:
                _launch_response(websocket, session, event.audio, t_commit,
                                 silence_secs=event.silence_secs)
            else:
                await websocket.send_json({"type": "state", "state": "listening"})


@app.websocket("/ws")
@app.websocket("/voice/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle voice WebSocket connections."""
    # Check for API key in query params or headers
    api_key_str = websocket.query_params.get("api_key") or \
                  websocket.headers.get("x-api-key")
    
    api_key: Optional[APIKey] = None
    
    if settings.require_auth:
        if not api_key_str:
            await websocket.close(code=4001, reason="API key required")
            return
        
        api_key = token_manager.validate_key(api_key_str)
        if not api_key:
            await websocket.close(code=4002, reason="Invalid API key")
            return
        
        if not token_manager.check_rate_limit(api_key):
            await websocket.close(code=4003, reason="Rate limit exceeded")
            return
        
        logger.info(f"Client connected: {api_key.name} (tier={api_key.tier})")
    else:
        # Dev mode - allow all
        if api_key_str:
            api_key = token_manager.validate_key(api_key_str)
        logger.info("Client connected (auth disabled)")
    
    await websocket.accept()

    session = _SessionState()
    session.conn_backend = backend.clone_for_session() if backend else None

    audio_buffer = []
    is_listening = False

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg["type"] == "session_start":
                # Server-driven continuous mode: per-connection turn engine.
                #
                # Optional resume: a client that sends a persistent client_id
                # and reconnects within the grace window gets its parked
                # _SessionState back (conn_backend + conversation history
                # intact) instead of starting a fresh conversation.
                client_id = msg.get("client_id")
                if not isinstance(client_id, str) or not client_id.strip():
                    client_id = None

                _purge_expired_registry()
                resumed = False
                history_turns = 0
                if client_id:
                    entry = _SESSION_REGISTRY.pop(client_id, None)
                    if entry is not None:
                        resumed = True
                        session = entry["session"]
                        # Defensively cancel any stale response pipeline —
                        # the old websocket it was bound to is gone.
                        await _cancel_response_task(session)
                        session.response_task = None
                        if session.conn_backend is not None:
                            history_turns = (
                                len(session.conn_backend.conversation_history) // 2
                            )
                session.client_id = client_id

                overrides = msg.get("config")
                turn_config = TurnConfig.from_env(
                    overrides if isinstance(overrides, dict) else None
                )
                session.mode = "continuous"
                # Always a fresh TurnEngine: VAD state, turn timers and
                # config must not carry over across a reconnect — only the
                # AI conversation history (on conn_backend) does.
                session.engine = TurnEngine(config=turn_config)
                is_listening = False
                audio_buffer = []
                await websocket.send_json({
                    "type": "session_started",
                    "mode": "continuous",
                    "resumed": resumed,
                    "history_turns": history_turns,
                    "config": {
                        "start_threshold": turn_config.start_threshold,
                        "min_speech_frames": turn_config.min_speech_frames,
                        "min_silence_secs": turn_config.min_silence_secs,
                        "recheck_interval_secs": turn_config.recheck_interval_secs,
                        "patience_ceiling_secs": turn_config.patience_ceiling_secs,
                        "semantic_enabled": session.engine.semantic_active,
                        "barge_in_min_speech_secs": turn_config.barge_in_min_speech_secs,
                        "semantic_threshold": turn_config.semantic_threshold,
                        "barge_candidate_secs": turn_config.barge_candidate_secs,
                        "barge_gap_frames": turn_config.barge_gap_frames,
                        "barge_responding_threshold": turn_config.barge_responding_threshold,
                        "stop_phrases": turn_config.stop_phrase_list,
                    },
                })
                await websocket.send_json({"type": "state", "state": "listening"})
                logger.info("Continuous session started")

            elif msg["type"] == "start_listening":
                is_listening = True
                audio_buffer = []
                await websocket.send_json({"type": "listening_started"})
                logger.debug("Started listening")

            elif msg["type"] == "stop_listening":
                is_listening = False

                if session.mode == "continuous" and session.engine:
                    # Manual commit of whatever the engine has buffered.
                    events = session.engine.force_commit()
                    await _handle_turn_events(websocket, session, events)
                elif audio_buffer:
                    audio_data = np.concatenate(audio_buffer)
                    await run_turn(websocket, audio_data, session.conn_backend)

                audio_buffer = []
                await websocket.send_json({"type": "listening_stopped"})
                logger.debug("Stopped listening")

            elif msg["type"] == "audio":
                if session.mode == "continuous" and session.engine:
                    audio_np = _decode_audio(msg)
                    if len(audio_np) > 0:
                        events = session.engine.feed(audio_np, time.monotonic())
                        await _handle_turn_events(websocket, session, events)
                elif is_listening:
                    audio_np = _decode_audio(msg)
                    audio_buffer.append(audio_np)

                    # VAD check - notify client if speech detected
                    if vad and len(audio_np) > 0:
                        has_speech = vad.is_speech(audio_np)
                        await websocket.send_json({
                            "type": "vad_status",
                            "speech_detected": has_speech,
                        })

            elif msg["type"] == "interrupt":
                # On-screen backup interrupt: kill the current reply but keep
                # the session alive and listening. In LISTENING state it
                # force-commits whatever the engine has (tap = "answer now").
                cancelled = await _cancel_response_task(session)
                await websocket.send_json({"type": "tts_cancelled"})
                if session.engine:
                    if cancelled or session.engine.agent_responding:
                        session.engine.resolve_barge("cancel", time.monotonic())
                        session.engine.set_agent_responding(False)
                        await websocket.send_json({"type": "state", "state": "listening"})
                        logger.info("Manual interrupt (button)")
                        voice_log("manual_interrupt", context="agent_responding")
                    else:
                        events = session.engine.force_commit()
                        if events:
                            logger.info("Manual commit (button)")
                            voice_log("manual_interrupt", context="commit")
                            await _handle_turn_events(websocket, session, events)
                        else:
                            await websocket.send_json({"type": "state", "state": "listening"})
                else:
                    await websocket.send_json({"type": "state", "state": "listening"})

            elif msg["type"] == "playback_done":
                voice_log("playback_done", turn_id=msg.get("turn_id"))
                if session.playback_done is not None:
                    session.playback_done.set()

            elif msg["type"] == "session_stop":
                # Kill switch: cancel any in-flight response, drop buffered
                # turn audio, and leave continuous mode. Explicit stop means
                # the user wants a clean end, not a resumable session — drop
                # any parked registry entry and forget the client_id so a
                # later disconnect doesn't re-park it.
                cancelled = await _cancel_response_task(session)
                if session.engine:
                    session.engine.reset()
                if session.client_id:
                    _SESSION_REGISTRY.pop(session.client_id, None)
                    session.client_id = None
                session.mode = "ptt"
                session.engine = None
                is_listening = False
                audio_buffer = []
                await websocket.send_json({
                    "type": "session_stopped",
                    "cancelled_response": cancelled,
                })
                await websocket.send_json({"type": "state", "state": "idle"})
                logger.info("Session stopped (kill switch)")

            elif msg["type"] == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
    finally:
        await _cancel_response_task(session)
        session.response_task = None
        # Park for possible resume (no-op if this connection never adopted
        # a client_id — legacy clients keep today's fresh-state behavior).
        _park_session(session)


# Serve static files for client
client_dir = Path(__file__).parent.parent / "client"
if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )

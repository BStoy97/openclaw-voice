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
import time
from pathlib import Path
from typing import Optional

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
tts: Optional[ChatterboxTTS] = None
backend: Optional[AIBackend] = None
vad: Optional[VoiceActivityDetector] = None


@app.on_event("startup")
async def startup():
    """Initialize models on server start."""
    global stt, tts, backend, vad
    
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
) -> None:
    """One full response turn: STT → streamed LLM → sentence-level TTS.

    Used by both the legacy stop_listening path and continuous-mode commits.
    Cancellation-safe: a barge-in cancels the surrounding task at any await.
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

    if not transcript.strip():
        return

    if continuous:
        await websocket.send_json(_tag({"type": "state", "state": "thinking"}))

    full_response = ""
    sentence_buffer = ""
    t_llm_first: Optional[float] = None
    t_first_audio: Optional[float] = None

    async def speak(text: str) -> None:
        """Synthesize one sentence and stream it as audio_chunk messages."""
        nonlocal t_first_audio
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

    if continuous:
        t_done = time.monotonic()
        await websocket.send_json(_tag({
            "type": "turn_metrics",
            "stt_ms": round((t_transcript - t_commit) * 1000),
            "llm_ttft_ms": (
                round((t_llm_first - t_transcript) * 1000)
                if t_llm_first is not None else None
            ),
            "tts_first_chunk_ms": (
                round((t_first_audio - (t_llm_first or t_transcript)) * 1000)
                if t_first_audio is not None else None
            ),
            "total_ms": round((t_done - t_commit) * 1000),
        }))


class _SessionState:
    """Per-WebSocket-connection state."""

    def __init__(self):
        self.mode = "ptt"  # "ptt" (legacy) or "continuous" (server-driven)
        self.engine: Optional[TurnEngine] = None
        self.conn_backend: Optional[AIBackend] = None
        self.response_task: Optional[asyncio.Task] = None
        self.turn_counter = 0
        self.current_turn_id: Optional[int] = None


async def _cancel_response_task(session: _SessionState) -> bool:
    """Cancel the in-flight response pipeline, if any. Returns True if one
    was actually cancelled."""
    task = session.response_task
    if task is None or task.done():
        return False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as e:  # noqa: BLE001 - task errors already logged there
        logger.debug(f"Cancelled response task raised: {e}")
    return True


def _launch_response(
    websocket: WebSocket, session: _SessionState, audio: np.ndarray, t_commit: float
) -> None:
    """Start the response pipeline as a cancellable background task."""
    turn_id = session.current_turn_id
    engine = session.engine
    engine.set_agent_responding(True)

    async def _task():
        try:
            await run_turn(
                websocket,
                audio,
                session.conn_backend,
                turn_id=turn_id,
                continuous=True,
                t_commit=t_commit,
            )
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
        elif event.type == BARGE_IN:
            logger.info("Barge-in: cancelling agent response")
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
                _launch_response(websocket, session, event.audio, t_commit)
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
                overrides = msg.get("config")
                turn_config = TurnConfig.from_env(
                    overrides if isinstance(overrides, dict) else None
                )
                session.mode = "continuous"
                session.engine = TurnEngine(config=turn_config)
                is_listening = False
                audio_buffer = []
                await websocket.send_json({
                    "type": "session_started",
                    "mode": "continuous",
                    "config": {
                        "start_threshold": turn_config.start_threshold,
                        "min_speech_frames": turn_config.min_speech_frames,
                        "min_silence_secs": turn_config.min_silence_secs,
                        "recheck_interval_secs": turn_config.recheck_interval_secs,
                        "patience_ceiling_secs": turn_config.patience_ceiling_secs,
                        "semantic_enabled": session.engine.semantic_active,
                        "barge_in_min_speech_secs": turn_config.barge_in_min_speech_secs,
                        "semantic_threshold": turn_config.semantic_threshold,
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

            elif msg["type"] == "session_stop":
                # Kill switch: cancel any in-flight response, drop buffered
                # turn audio, and leave continuous mode.
                cancelled = await _cancel_response_task(session)
                if session.engine:
                    session.engine.reset()
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

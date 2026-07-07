"""
Server integration tests for OpenClaw Voice.
"""

import pytest
import asyncio
import json
import base64
import numpy as np
import os
import shutil
import sys
import subprocess
import time
import socket
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPER_VOICE = REPO_ROOT / "models" / "piper" / "en_US-amy-medium.onnx"


def generate_speech_16k(text: str):
    """Generate real speech (16 kHz float32) with the local piper voice, so
    the server's real Silero VAD detects it. Returns None if unavailable."""
    piper_bin = shutil.which("piper")
    if not piper_bin or not PIPER_VOICE.exists():
        return None
    try:
        raw = subprocess.run(
            [piper_bin, "-m", str(PIPER_VOICE),
             "-c", str(PIPER_VOICE) + ".json", "--output-raw"],
            input=(text + "\n").encode(),
            capture_output=True,
            timeout=60,
        ).stdout
        if not raw:
            return None
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        from scipy.signal import resample_poly
        return resample_poly(audio, 16000, 22050).astype(np.float32)
    except Exception:
        return None


def audio_msg(audio: np.ndarray) -> str:
    return json.dumps({
        "type": "audio",
        "data": base64.b64encode(audio.astype(np.float32).tobytes()).decode(),
    })


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


@pytest.fixture(scope="module")
def server():
    """Start the server for testing."""
    port = 8799  # Use high port to avoid conflicts
    
    # Skip if port already in use
    if is_port_in_use(port):
        pytest.skip(f"Port {port} already in use")
    
    # Start server
    env = os.environ.copy()
    env['OPENCLAW_PORT'] = str(port)
    env['OPENCLAW_STT_MODEL'] = 'tiny'  # Use tiny for fast tests
    env['OPENCLAW_REQUIRE_AUTH'] = 'false'
    # Hermetic: server must boot keyless into echo mode (no real credentials
    # available or required for these tests).
    for key in (
        'OPENAI_API_KEY',
        'OPENCLAW_GATEWAY_URL',
        'OPENCLAW_GATEWAY_TOKEN',
    ):
        env.pop(key, None)

    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'src.server.main:app',
         '--host', '127.0.0.1', '--port', str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    # Wait for server to be ready
    max_wait = 60
    for i in range(max_wait):
        if is_port_in_use(port):
            break
        time.sleep(1)
    else:
        proc.terminate()
        pytest.fail("Server did not start in time")
    
    yield f"ws://127.0.0.1:{port}/ws", f"http://127.0.0.1:{port}"
    
    # Cleanup
    proc.terminate()
    proc.wait(timeout=5)


class TestServerHTTP:
    """Test HTTP endpoints."""
    
    def test_index_page(self, server):
        """Test that index page loads."""
        import httpx
        
        ws_url, http_url = server
        response = httpx.get(f"{http_url}/")
        
        assert response.status_code == 200
        assert "OpenClaw Voice" in response.text
        assert "voice-button" in response.text


class TestServerWebSocket:
    """Test WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_connect(self, server):
        """Test WebSocket connection."""
        import websockets
        
        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            # Connection successful if we get here
            assert ws is not None
    
    @pytest.mark.asyncio
    async def test_ping_pong(self, server):
        """Test ping/pong."""
        import websockets
        
        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_start_stop_listening(self, server):
        """Test start/stop listening cycle."""
        import websockets
        
        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            # Start
            await ws.send(json.dumps({"type": "start_listening"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "listening_started"
            
            # Stop
            await ws.send(json.dumps({"type": "stop_listening"}))
            response = json.loads(await ws.recv())
            assert response["type"] == "listening_stopped"
    
    @pytest.mark.asyncio
    async def test_audio_flow(self, server):
        """Test sending audio and getting response."""
        import websockets
        
        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            # Start listening
            await ws.send(json.dumps({"type": "start_listening"}))
            await ws.recv()  # listening_started
            
            # Send some audio (silence)
            audio = np.zeros(16000, dtype=np.float32)
            audio_b64 = base64.b64encode(audio.tobytes()).decode()
            
            await ws.send(json.dumps({
                "type": "audio",
                "data": audio_b64,
            }))
            
            # Stop listening
            await ws.send(json.dumps({"type": "stop_listening"}))
            
            # Should get transcript first, then listening_stopped
            messages = []
            for _ in range(5):  # Collect up to 5 messages
                try:
                    response = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                    messages.append(response["type"])
                    if response["type"] == "listening_stopped":
                        break
                except asyncio.TimeoutError:
                    break
            
            # Should have gotten transcript and/or listening_stopped
            assert "transcript" in messages or "listening_stopped" in messages


class TestContinuousMode:
    """Server-driven turn detection over the WebSocket protocol."""

    @pytest.mark.asyncio
    async def test_session_start_enters_listening(self, server):
        """session_start → session_started (with effective config) + state listening."""
        import websockets

        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "session_start",
                "mode": "continuous",
                "config": {"patience_ceiling_secs": 50},  # must be clamped
            }))
            started = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            assert started["type"] == "session_started"
            assert started["mode"] == "continuous"
            assert started["config"]["patience_ceiling_secs"] <= 20.0

            state = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert state == {"type": "state", "state": "listening"}

    @pytest.mark.asyncio
    async def test_legacy_ptt_still_works_alongside(self, server):
        """A connection that never sends session_start stays in PTT mode."""
        import websockets

        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"type": "start_listening"}))
            response = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert response["type"] == "listening_started"
            await ws.send(json.dumps({"type": "stop_listening"}))
            response = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert response["type"] == "listening_stopped"

    @pytest.mark.asyncio
    async def test_speech_then_silence_commits_turn(self, server):
        """Real speech then silence → turn_started / turn_committed /
        transcript / response / turn_metrics protocol flow."""
        import websockets

        speech = generate_speech_16k(
            "Hello there, can you hear me? I have a quick question."
        )
        if speech is None:
            pytest.skip("piper binary or voice model unavailable")

        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "session_start",
                "mode": "continuous",
                # Tight timings so the test completes quickly.
                "config": {
                    "min_silence_secs": 0.6,
                    "recheck_interval_secs": 0.5,
                    "patience_ceiling_secs": 5.0,
                },
            }))

            seen = {}          # type -> first message of that type
            order = []
            silence = np.zeros(4096, dtype=np.float32)

            # Stream the speech in ScriptProcessor-sized chunks.
            for i in range(0, len(speech), 4096):
                await ws.send(audio_msg(speech[i:i + 4096]))
                await asyncio.sleep(0.005)

            deadline = time.time() + 60
            committed = False
            while time.time() < deadline:
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.1))
                    if msg["type"] not in seen:
                        seen[msg["type"]] = msg
                    order.append(msg["type"])
                    if msg["type"] == "turn_committed":
                        committed = True
                    if msg["type"] == "turn_metrics":
                        break
                except asyncio.TimeoutError:
                    pass
                if not committed:
                    # Keep the audio stream flowing (silence) so the engine
                    # can measure the pause, as a real client would.
                    await ws.send(audio_msg(silence))

            assert "turn_started" in seen, f"messages seen: {order}"
            assert "turn_committed" in seen, f"messages seen: {order}"
            assert seen["turn_committed"]["reason"] in ("semantic", "ceiling", "timeout")
            assert seen["turn_committed"]["turn_id"] == 1
            assert "transcript" in seen, f"messages seen: {order}"
            assert seen["transcript"]["turn_id"] == 1
            # Whisper tiny on synthetic speech: content varies, but the
            # pipeline (echo backend) must complete and report metrics.
            assert "response_complete" in seen, f"messages seen: {order}"
            assert "turn_metrics" in seen, f"messages seen: {order}"
            metrics = seen["turn_metrics"]
            assert metrics["turn_id"] == 1
            assert metrics["stt_ms"] >= 0
            assert metrics["total_ms"] >= metrics["stt_ms"]
            # audio chunks must carry the backend's real sample rate
            if "audio_chunk" in seen:
                assert seen["audio_chunk"]["sample_rate"] > 0

            # After the response, the server returns to listening.
            state_after = None
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.2))
                    if msg["type"] == "state" and msg["state"] == "listening":
                        state_after = msg
                        break
                except asyncio.TimeoutError:
                    pass
            assert state_after is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

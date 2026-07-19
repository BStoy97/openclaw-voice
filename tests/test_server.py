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
        assert "OC Voice" in response.text
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
            # Barge-in tuning (transcribe-then-decide) is reported too.
            assert started["config"]["barge_candidate_secs"] > 0
            assert started["config"]["barge_gap_frames"] >= 1
            assert 0 <= started["config"]["barge_responding_threshold"] <= 1
            assert "stop" in started["config"]["stop_phrases"]

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


class TestStripTrailingKeyword:
    def test_strips_variants(self):
        from src.server.main import strip_trailing_keyword as stk
        assert stk("Check the weather, over", "over") == "Check the weather"
        assert stk("Check the weather. Over.", "over") == "Check the weather"
        assert stk("Is the meeting over", "over") == "Is the meeting"
        assert stk("Game over man, game over!", "over") == "Game over man, game"
        assert stk("No keyword here", "over") == "No keyword here"
        assert stk("over", "over") == ""
        assert stk("Push it over the line", "over") == "Push it over the line"
        assert stk("anything", "") == "anything"


class TestManualInterrupt:
    """The on-screen backup interrupt: {"type": "interrupt"}."""

    @pytest.mark.asyncio
    async def test_interrupt_in_listening_state_acks(self, server):
        import websockets

        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"type": "session_start", "mode": "continuous"}))
            await ws.recv()  # session_started
            await ws.recv()  # state listening
            await ws.send(json.dumps({"type": "interrupt"}))
            got = []
            for _ in range(3):
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                except asyncio.TimeoutError:
                    break
                got.append(msg["type"])
                if msg["type"] == "state":
                    break
            assert "tts_cancelled" in got  # always flushes client playback
            assert "state" in got

    @pytest.mark.asyncio
    async def test_interrupt_without_session_is_safe(self, server):
        import websockets

        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"type": "interrupt"}))
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert msg["type"] == "tts_cancelled"
            # connection stays usable
            await ws.send(json.dumps({"type": "ping"}))
            for _ in range(3):
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if msg["type"] == "pong":
                    break
            assert msg["type"] == "pong"


class TestSessionResume:
    """Reconnect session continuity: client_id resume registry."""

    @pytest.mark.asyncio
    async def test_first_connect_not_resumed(self, server):
        """A brand-new client_id never seen before -> resumed:false."""
        import websockets

        ws_url, _ = server
        client_id = "resume-test-fresh"
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "session_start",
                "mode": "continuous",
                "client_id": client_id,
            }))
            started = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            assert started["type"] == "session_started"
            assert started["resumed"] is False
            assert started["history_turns"] == 0
            await asyncio.wait_for(ws.recv(), timeout=5)  # state: listening
            # Explicitly stop so this client_id doesn't leak into the
            # registry and affect other tests in this module.
            await ws.send(json.dumps({"type": "session_stop"}))
            await asyncio.wait_for(ws.recv(), timeout=5)

    @pytest.mark.asyncio
    async def test_reconnect_same_client_id_resumes(self, server):
        """Disconnect (without session_stop) then reconnect with the same
        client_id within the grace window -> resumed:true."""
        import websockets

        ws_url, _ = server
        client_id = "resume-test-reconnect"

        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "session_start",
                "mode": "continuous",
                "client_id": client_id,
            }))
            started = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            assert started["resumed"] is False
        # `async with` exit closes the socket -> server parks the session.

        # Give the server a moment to process the disconnect and park.
        await asyncio.sleep(0.5)

        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "session_start",
                "mode": "continuous",
                "client_id": client_id,
            }))
            started = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            assert started["type"] == "session_started"
            assert started["resumed"] is True
            await asyncio.wait_for(ws.recv(), timeout=5)  # state: listening

            await ws.send(json.dumps({"type": "session_stop"}))
            await asyncio.wait_for(ws.recv(), timeout=5)

    @pytest.mark.asyncio
    async def test_reconnect_different_client_id_not_resumed(self, server):
        """A different client_id never resumes another session's state,
        even though the first session_start's client_id remains parked."""
        import websockets

        ws_url, _ = server
        client_id_a = "resume-test-diff-a"
        client_id_b = "resume-test-diff-b"

        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "session_start",
                "mode": "continuous",
                "client_id": client_id_a,
            }))
            await asyncio.wait_for(ws.recv(), timeout=15)
        # `async with` exit closes the socket -> client_id_a gets parked.

        await asyncio.sleep(0.5)

        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "session_start",
                "mode": "continuous",
                "client_id": client_id_b,
            }))
            started = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            assert started["resumed"] is False
            await asyncio.wait_for(ws.recv(), timeout=5)  # state: listening

            # Clean up so client_id_b/a don't leak into other tests.
            await ws.send(json.dumps({"type": "session_stop"}))
            await asyncio.wait_for(ws.recv(), timeout=5)

    @pytest.mark.asyncio
    async def test_session_stop_clears_registry_for_resume(self, server):
        """session_stop (kill switch) removes the client_id entry, so a
        later reconnect with the same id is NOT resumed."""
        import websockets

        ws_url, _ = server
        client_id = "resume-test-stopped"

        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "session_start",
                "mode": "continuous",
                "client_id": client_id,
            }))
            await asyncio.wait_for(ws.recv(), timeout=15)  # session_started
            await asyncio.wait_for(ws.recv(), timeout=5)   # state: listening

            await ws.send(json.dumps({"type": "session_stop"}))
            stopped = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert stopped["type"] == "session_stopped"
        # Socket closes after an explicit session_stop -> must NOT be
        # re-parked (client_id was forgotten by session_stop).

        await asyncio.sleep(0.5)

        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "session_start",
                "mode": "continuous",
                "client_id": client_id,
            }))
            started = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            assert started["resumed"] is False
            await asyncio.wait_for(ws.recv(), timeout=5)  # state: listening
            await ws.send(json.dumps({"type": "session_stop"}))
            await asyncio.wait_for(ws.recv(), timeout=5)

    @pytest.mark.asyncio
    async def test_legacy_session_start_without_client_id(self, server):
        """No client_id at all -> legacy behavior, resumed:false, and the
        connection never touches the resume registry."""
        import websockets

        ws_url, _ = server
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({
                "type": "session_start",
                "mode": "continuous",
            }))
            started = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            assert started["resumed"] is False
            assert started["history_turns"] == 0


class TestDecideBarge:
    """Unit tests for the transcribe-then-decide barge classifier
    (src.server.main.decide_barge), run in-process — the async candidate
    handler is a thin wrapper around this pure function."""

    @staticmethod
    def _decide(text, stop_phrases=None):
        from src.server.main import decide_barge
        return decide_barge(text, stop_phrases)

    # --- stop phrases -> cancel ------------------------------------------

    def test_stop_word_alone(self):
        assert self._decide("stop") == "cancel"

    def test_stop_with_punctuation_and_case(self):
        assert self._decide("Stop!") == "cancel"
        assert self._decide("STOP.") == "cancel"

    def test_stop_phrase_inside_sentence(self):
        assert self._decide("please stop talking right now") == "cancel"
        assert self._decide("no no wait a second") == "cancel"

    def test_multiword_stop_phrases(self):
        assert self._decide("hold on") == "cancel"
        assert self._decide("Hold on, hold on!") == "cancel"
        assert self._decide("okay stop") == "cancel"
        assert self._decide("that's enough") == "cancel"
        assert self._decide("never mind") == "cancel"
        assert self._decide("nevermind") == "cancel"

    def test_stop_phrase_beats_takeover_word_count(self):
        # 5 words, but contains a stop phrase -> cancel wins.
        assert self._decide("hey can you please stop") == "cancel"

    # --- word boundaries ---------------------------------------------------

    def test_stopwatch_does_not_match_stop(self):
        # "stopwatch" must NOT trigger the "stop" phrase; 6 words -> takeover
        assert self._decide("I checked my stopwatch yesterday morning") == "takeover"

    def test_single_word_stopwatch_ignored(self):
        assert self._decide("stopwatch") == "ignore"

    def test_waiter_does_not_match_wait(self):
        assert self._decide("the waiter brought our food") == "takeover"

    # --- takeover ------------------------------------------------------------

    def test_three_plus_words_takeover(self):
        assert self._decide("tell me more") == "takeover"
        assert self._decide("what about the weather") == "takeover"

    # --- ignore ---------------------------------------------------------------

    def test_empty_and_whitespace_ignored(self):
        assert self._decide("") == "ignore"
        assert self._decide("   ") == "ignore"
        assert self._decide(None) == "ignore"

    def test_short_filler_ignored(self):
        assert self._decide("uh") == "ignore"
        assert self._decide("um yeah") == "ignore"
        assert self._decide("Hmm...") == "ignore"

    # --- custom per-session stop phrases -------------------------------------

    def test_custom_stop_phrases(self):
        assert self._decide("banana", ["banana"]) == "cancel"
        # Default phrases no longer apply when a custom list is given.
        assert self._decide("stop", ["banana"]) == "ignore"
        assert self._decide("red light", ["red light"]) == "cancel"


class TestSessionRegistryUnit:
    """Direct unit tests of the resume registry's purge/cap logic, run
    in-process against src.server.main (no server/websocket needed)."""

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        from src.server import main as server_main
        server_main._SESSION_REGISTRY.clear()
        yield
        server_main._SESSION_REGISTRY.clear()

    def test_grace_secs_default_and_clamp(self, monkeypatch):
        from src.server import main as server_main

        monkeypatch.delenv("OPENCLAW_SESSION_GRACE_SECS", raising=False)
        assert server_main._session_grace_secs() == 600.0

        monkeypatch.setenv("OPENCLAW_SESSION_GRACE_SECS", "-5")
        assert server_main._session_grace_secs() == 0.0

        monkeypatch.setenv("OPENCLAW_SESSION_GRACE_SECS", "999999")
        assert server_main._session_grace_secs() == 3600.0

        monkeypatch.setenv("OPENCLAW_SESSION_GRACE_SECS", "45")
        assert server_main._session_grace_secs() == 45.0

        monkeypatch.setenv("OPENCLAW_SESSION_GRACE_SECS", "not-a-number")
        assert server_main._session_grace_secs() == 600.0

    def test_purge_expired_removes_only_expired(self):
        from src.server import main as server_main

        reg = server_main._SESSION_REGISTRY
        reg["alive"] = {"session": object(), "expires_at": 1000.0}
        reg["dead_a"] = {"session": object(), "expires_at": 10.0}
        reg["dead_b"] = {"session": object(), "expires_at": 50.0}

        server_main._purge_expired_registry(now=100.0)

        assert set(reg.keys()) == {"alive"}

    def test_purge_expired_boundary_is_expired(self):
        """expires_at == now counts as expired (<=)."""
        from src.server import main as server_main

        reg = server_main._SESSION_REGISTRY
        reg["exact"] = {"session": object(), "expires_at": 100.0}

        server_main._purge_expired_registry(now=100.0)

        assert "exact" not in reg

    def test_cap_drops_oldest_first(self):
        from src.server import main as server_main

        reg = server_main._SESSION_REGISTRY
        for i in range(server_main.SESSION_REGISTRY_CAP + 5):
            reg[f"client-{i}"] = {"session": object(), "expires_at": float(i)}

        assert len(reg) == server_main.SESSION_REGISTRY_CAP + 5
        server_main._cap_session_registry()

        assert len(reg) == server_main.SESSION_REGISTRY_CAP
        # The 5 oldest (lowest-indexed, inserted first) must be gone.
        for i in range(5):
            assert f"client-{i}" not in reg
        # The most recently inserted entries survive.
        for i in range(5, server_main.SESSION_REGISTRY_CAP + 5):
            assert f"client-{i}" in reg

    def test_park_session_noop_without_client_id(self):
        from src.server import main as server_main

        session = server_main._SessionState()
        session.client_id = None
        server_main._park_session(session)
        assert len(server_main._SESSION_REGISTRY) == 0

    def test_park_session_registers_with_expiry(self, monkeypatch):
        from src.server import main as server_main

        monkeypatch.setenv("OPENCLAW_SESSION_GRACE_SECS", "10")
        session = server_main._SessionState()
        session.client_id = "unit-park-test"

        before = time.monotonic()
        server_main._park_session(session)
        after = time.monotonic()

        entry = server_main._SESSION_REGISTRY["unit-park-test"]
        assert entry["session"] is session
        assert before + 10 <= entry["expires_at"] <= after + 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

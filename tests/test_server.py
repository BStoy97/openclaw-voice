"""
Server integration tests for OpenClaw Voice.
"""

import pytest
import asyncio
import json
import base64
import numpy as np
import os
import sys
import subprocess
import time
import socket

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

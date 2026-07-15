"""
Unit + integration tests for src/server/tts.py (TTSEngine).

Design goals under test:
  - default backend ladder is piper (local) -> elevenlabs (opt-in only) -> mock
  - ElevenLabs is only ever selected when explicitly requested via
    OPENCLAW_TTS_BACKEND=elevenlabs (having just the API key is not enough)
  - synthesize_stream() always yields raw int16 PCM bytes, never float32,
    regardless of backend
  - piper is driven via asyncio subprocess (mocked here so CI doesn't need
    the real binary) with a real end-to-end test gated on the binary existing
"""

import asyncio
import json
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.server.tts import TTSEngine, ChatterboxTTS, MOCK_SAMPLE_RATE, ELEVENLABS_SAMPLE_RATE


# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def clean_tts_env(monkeypatch):
    """Every test starts from a clean slate for the env vars TTSEngine reads."""
    for var in (
        "OPENCLAW_TTS_BACKEND",
        "OPENCLAW_TTS_VOICE",
        "OPENCLAW_TTS_PIPER_BIN",
        "ELEVENLABS_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def fake_piper_voice(tmp_path):
    """Writes a fake .onnx + .onnx.json pair and points OPENCLAW_TTS_VOICE at it."""
    def _make(sample_rate=22050):
        model_path = tmp_path / "fakevoice.onnx"
        config_path = tmp_path / "fakevoice.onnx.json"
        model_path.write_bytes(b"not-a-real-model")
        config_path.write_text(json.dumps({"audio": {"sample_rate": sample_rate}}))
        return str(model_path)
    return _make


@pytest.fixture
def fake_piper_bin(tmp_path):
    """A file that exists on disk so shutil.which()/isfile() selection succeeds."""
    bin_path = tmp_path / "fake-piper"
    bin_path.write_text("#!/bin/sh\n")
    bin_path.chmod(0o755)
    return str(bin_path)


class _FakeStdin:
    def __init__(self):
        self.written = b""
        self.closed = False

    def write(self, data):
        self.written += data

    async def drain(self):
        pass

    def close(self):
        self.closed = True


class _FakeStdout:
    def __init__(self, data: bytes, chunk_size: int = 8192):
        self._chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        if not self._chunks:
            self._chunks = [b""]
        self._idx = 0

    async def read(self, n=-1):
        if self._idx >= len(self._chunks):
            return b""
        chunk = self._chunks[self._idx]
        self._idx += 1
        return chunk


class _FakeStderr:
    async def read(self):
        return b""


class _FakeProcess:
    def __init__(self, data: bytes, returncode: int = 0):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(data)
        self.stderr = _FakeStderr()
        self._returncode = returncode
        self.returncode = None  # None until wait(), like asyncio.subprocess
        self.killed = False

    def kill(self):
        self.killed = True
        if self._returncode == 0:
            self._returncode = -9

    async def wait(self):
        self.returncode = self._returncode
        return self._returncode


def _patch_subprocess(monkeypatch, audio_bytes: bytes, returncode: int = 0):
    """Replace asyncio.create_subprocess_exec so no real piper binary runs."""
    captured = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _FakeProcess(audio_bytes, returncode=returncode)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    return captured


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #


def test_default_backend_is_piper(monkeypatch, fake_piper_voice, fake_piper_bin):
    monkeypatch.setenv("OPENCLAW_TTS_PIPER_BIN", fake_piper_bin)
    monkeypatch.setenv("OPENCLAW_TTS_VOICE", fake_piper_voice(sample_rate=22050))

    engine = TTSEngine()

    assert engine._backend == "piper"
    assert engine.sample_rate == 22050


def test_falls_back_to_mock_when_piper_binary_missing(monkeypatch):
    monkeypatch.setenv("OPENCLAW_TTS_PIPER_BIN", "/definitely/not/a/real/piper/binary")

    engine = TTSEngine()

    assert engine._backend == "mock"
    assert engine.sample_rate == MOCK_SAMPLE_RATE


def test_falls_back_to_mock_when_voice_files_missing(monkeypatch, fake_piper_bin):
    monkeypatch.setenv("OPENCLAW_TTS_PIPER_BIN", fake_piper_bin)
    monkeypatch.setenv("OPENCLAW_TTS_VOICE", "/no/such/voice.onnx")

    engine = TTSEngine()

    assert engine._backend == "mock"


def test_elevenlabs_key_alone_does_not_select_elevenlabs(monkeypatch):
    """Owner decision: ElevenLabs is opt-in only. Just having the key must not
    switch the backend away from local/free piper (or mock, if piper is
    unavailable in this environment)."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-fake-key")
    # No OPENCLAW_TTS_BACKEND set -> defaults to piper ladder, never elevenlabs.

    engine = TTSEngine()

    assert engine._backend != "elevenlabs"


def test_elevenlabs_selected_when_explicit_and_sdk_available(monkeypatch):
    monkeypatch.setenv("OPENCLAW_TTS_BACKEND", "elevenlabs")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-fake-key")

    class _FakeElevenLabs:
        def __init__(self, api_key):
            self.api_key = api_key

    import elevenlabs
    monkeypatch.setattr(elevenlabs, "ElevenLabs", _FakeElevenLabs)

    engine = TTSEngine()

    assert engine._backend == "elevenlabs"
    assert engine.sample_rate == ELEVENLABS_SAMPLE_RATE


def test_elevenlabs_explicit_but_no_key_falls_back(monkeypatch, fake_piper_voice, fake_piper_bin):
    monkeypatch.setenv("OPENCLAW_TTS_BACKEND", "elevenlabs")
    monkeypatch.setenv("OPENCLAW_TTS_PIPER_BIN", fake_piper_bin)
    monkeypatch.setenv("OPENCLAW_TTS_VOICE", fake_piper_voice())

    engine = TTSEngine()

    assert engine._backend != "elevenlabs"
    assert engine._backend == "piper"


def test_explicit_mock_backend(monkeypatch):
    monkeypatch.setenv("OPENCLAW_TTS_BACKEND", "mock")

    engine = TTSEngine()

    assert engine._backend == "mock"
    assert engine.sample_rate == MOCK_SAMPLE_RATE


def test_chatterbox_alias_is_tts_engine():
    """main.py still imports `ChatterboxTTS` unchanged."""
    assert ChatterboxTTS is TTSEngine


def test_no_runtime_pip_install(monkeypatch):
    """Regression guard: the old code shelled out to `pip install elevenlabs`
    on ImportError. Assert subprocess.check_call is never invoked by TTSEngine
    even when elevenlabs backend is requested and the SDK import is broken."""
    import builtins

    monkeypatch.setenv("OPENCLAW_TTS_BACKEND", "elevenlabs")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-fake-key")

    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "elevenlabs":
            raise ImportError("simulated: elevenlabs not installed")
        return real_import(name, *args, **kwargs)

    calls = []
    monkeypatch.setattr(builtins, "__import__", blocking_import)

    import subprocess
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: calls.append((a, k)))

    engine = TTSEngine()

    assert engine._backend != "elevenlabs"
    assert calls == []  # pip install was never called


# --------------------------------------------------------------------------- #
# Wire format: int16 everywhere, never float32
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mock_stream_yields_int16_bytes(monkeypatch):
    monkeypatch.setenv("OPENCLAW_TTS_BACKEND", "mock")
    engine = TTSEngine()

    chunks = [c async for c in engine.synthesize_stream("hello there")]
    assert len(chunks) == 1
    data = chunks[0]

    n_samples = int(engine.sample_rate * 0.5)
    # int16 = 2 bytes/sample; if this were float32 (4 bytes/sample) the length
    # would be double.
    assert len(data) == n_samples * 2
    arr = np.frombuffer(data, dtype=np.int16)
    assert arr.dtype == np.int16
    assert np.all(arr == 0)


@pytest.mark.asyncio
async def test_mock_synthesize_returns_float32_numpy_but_wire_is_int16(monkeypatch):
    """synthesize() (non-streaming, in-process convenience) still returns
    float32 for numeric compatibility with callers doing math on it, but it
    must be derived from int16 wire bytes, not carry float32 over the wire."""
    monkeypatch.setenv("OPENCLAW_TTS_BACKEND", "mock")
    engine = TTSEngine()

    audio = await engine.synthesize("hello")
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0
    assert np.all(np.abs(audio) <= 1.0)


@pytest.mark.asyncio
async def test_piper_stream_yields_int16_passthrough(monkeypatch, fake_piper_voice, fake_piper_bin):
    monkeypatch.setenv("OPENCLAW_TTS_PIPER_BIN", fake_piper_bin)
    monkeypatch.setenv("OPENCLAW_TTS_VOICE", fake_piper_voice(sample_rate=22050))

    engine = TTSEngine()
    assert engine._backend == "piper"

    fake_samples = np.array([100, -100, 200, -200, 0, 32000], dtype=np.int16)
    fake_audio_bytes = fake_samples.tobytes()
    captured = _patch_subprocess(monkeypatch, fake_audio_bytes)

    chunks = [c async for c in engine.synthesize_stream("hello")]
    data = b"".join(chunks)

    assert data == fake_audio_bytes
    assert len(data) % 2 == 0  # int16-aligned
    roundtrip = np.frombuffer(data, dtype=np.int16)
    assert np.array_equal(roundtrip, fake_samples)

    # subprocess was invoked with the piper CLI and --output-raw, text on stdin
    args = captured["args"]
    assert args[0] == engine._piper_bin
    assert "--output-raw" in args
    assert "-m" in args and str(engine._piper_model_path) in args
    assert "-c" in args and str(engine._piper_config_path) in args


@pytest.mark.asyncio
async def test_piper_stream_reads_in_chunks_as_they_arrive(monkeypatch, fake_piper_voice, fake_piper_bin):
    """Real streaming: multiple stdout reads should produce multiple yielded
    chunks rather than buffering everything into one blob."""
    monkeypatch.setenv("OPENCLAW_TTS_PIPER_BIN", fake_piper_bin)
    monkeypatch.setenv("OPENCLAW_TTS_VOICE", fake_piper_voice())

    engine = TTSEngine()
    big_audio = np.zeros(20000, dtype=np.int16).tobytes()  # > one 8KB chunk
    _patch_subprocess(monkeypatch, big_audio)

    chunks = [c async for c in engine.synthesize_stream("hello world")]
    assert len(chunks) > 1
    assert b"".join(chunks) == big_audio
    for c in chunks:
        assert isinstance(c, (bytes, bytearray))


@pytest.mark.asyncio
async def test_elevenlabs_stream_runs_off_event_loop(monkeypatch):
    """The sync SDK generator must run via a worker thread (asyncio.to_thread /
    run_in_executor), not block the event loop directly."""
    monkeypatch.setenv("OPENCLAW_TTS_BACKEND", "elevenlabs")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-fake-key")

    thread_ids = []

    class _FakeConvertGenerator:
        def __iter__(self):
            import threading
            thread_ids.append(threading.get_ident())
            for chunk in [b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"]:
                yield chunk

    class _FakeTextToSpeech:
        def convert(self, **kwargs):
            return _FakeConvertGenerator()

    class _FakeElevenLabsClient:
        def __init__(self, api_key):
            self.text_to_speech = _FakeTextToSpeech()

    class _FakeElevenLabs:
        def __new__(cls, api_key):
            return _FakeElevenLabsClient(api_key)

    import elevenlabs
    monkeypatch.setattr(elevenlabs, "ElevenLabs", _FakeElevenLabs)

    engine = TTSEngine()
    assert engine._backend == "elevenlabs"

    import threading
    main_thread_id = threading.get_ident()

    chunks = [c async for c in engine.synthesize_stream("hello")]
    assert b"".join(chunks) == b"\x01\x00\x02\x00\x03\x00\x04\x00"
    assert thread_ids, "convert() generator was never consumed"
    assert thread_ids[0] != main_thread_id  # ran off the event loop thread


@pytest.mark.asyncio
async def test_synthesize_stream_empty_text_yields_nothing(monkeypatch):
    monkeypatch.setenv("OPENCLAW_TTS_BACKEND", "mock")
    engine = TTSEngine()
    chunks = [c async for c in engine.synthesize_stream("   ")]
    assert chunks == []


# --------------------------------------------------------------------------- #
# Real integration test (only runs if the piper CLI is actually installed)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not shutil.which("piper"), reason="piper CLI not installed")
@pytest.mark.asyncio
async def test_real_piper_synthesizes_hello():
    """End-to-end: real piper binary + real downloaded voice model."""
    engine = TTSEngine()
    assert engine._backend == "piper", (
        "piper binary is on PATH but engine didn't select it - "
        "check models/piper/ has the voice files (see scripts/download_models.py piper)"
    )

    chunks = [c async for c in engine.synthesize_stream("hello")]
    audio_bytes = b"".join(chunks)

    assert len(audio_bytes) > 0
    assert len(audio_bytes) % 2 == 0  # valid int16 stream
    assert engine.sample_rate > 0

    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    assert len(samples) > 0
    # Real speech audio shouldn't be pure silence.
    assert np.abs(samples.astype(np.int64)).max() > 0

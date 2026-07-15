"""
Text-to-Speech module: local-first Piper backend with an opt-in ElevenLabs
premium tier.

Backend ladder (selected via OPENCLAW_TTS_BACKEND, default "piper"):
  1. piper      - local, free, DEFAULT. Female en_US voice (amy) by default.
                  Invokes the `piper` CLI as a subprocess; no local ML deps.
  2. elevenlabs - cloud, opt-in premium. Only used when the caller explicitly
                  sets OPENCLAW_TTS_BACKEND=elevenlabs *and* ELEVENLABS_API_KEY
                  is present. Never runtime-pip-installs; if the SDK isn't
                  importable we log and fall through.
  3. mock       - silence fallback so the pipeline never crashes without a
                  configured backend.

Every backend's synthesize_stream() yields raw **int16 PCM** bytes ("s16le")
so the wire format always matches what the browser client decodes. There is
no float32-over-the-wire path left in this module.
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger

# models/piper/ relative to the repo root, resolved from this file's location
# so behavior doesn't depend on the process's current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPER_MODEL_DIR = _REPO_ROOT / "models" / "piper"
DEFAULT_PIPER_VOICE = "en_US-amy-medium"
DEFAULT_PIPER_BIN = "piper"

MOCK_SAMPLE_RATE = 24000
ELEVENLABS_SAMPLE_RATE = 24000

_STDOUT_CHUNK_SIZE = 8192


class TTSEngine:
    """Text-to-Speech engine: Piper (local, default) -> ElevenLabs (opt-in) -> mock."""

    def __init__(
        self,
        voice_sample: Optional[str] = None,  # unused by piper/elevenlabs; kept for API compat
        device: str = "auto",
        voice_id: Optional[str] = None,  # ElevenLabs voice ID
    ):
        self.voice_sample = voice_sample
        self.device = device
        self.voice_id = voice_id or "cgSgspJ2msm6clMCkdW9"  # Jessica (ElevenLabs only)

        self._backend = "mock"
        self._elevenlabs_client = None

        # Piper config (resolved during backend selection)
        self._piper_bin = os.environ.get("OPENCLAW_TTS_PIPER_BIN", DEFAULT_PIPER_BIN)
        self._piper_voice_name = os.environ.get("OPENCLAW_TTS_VOICE", DEFAULT_PIPER_VOICE)
        self._piper_model_path: Optional[Path] = None
        self._piper_config_path: Optional[Path] = None

        # Actual output sample rate of whichever backend gets selected. The
        # caller (main.py) should use this instead of a hardcoded constant.
        self.sample_rate = MOCK_SAMPLE_RATE

        self._select_backend()

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #

    def _select_backend(self) -> None:
        requested = os.environ.get("OPENCLAW_TTS_BACKEND", "piper").strip().lower()
        if requested not in ("piper", "elevenlabs", "mock"):
            logger.warning(f"Unknown OPENCLAW_TTS_BACKEND={requested!r}, defaulting to piper")
            requested = "piper"

        if requested == "mock":
            self._backend = "mock"
            self.sample_rate = MOCK_SAMPLE_RATE
            logger.info("TTS backend: mock (explicitly requested)")
            return

        if requested == "elevenlabs":
            if self._setup_elevenlabs():
                return
            logger.warning("OPENCLAW_TTS_BACKEND=elevenlabs but ElevenLabs unavailable; falling back to piper")

        # Default path, and elevenlabs-requested-but-unavailable fallback.
        if self._setup_piper():
            return

        logger.warning("No TTS backend available - using mock mode (silence)")
        self._backend = "mock"
        self.sample_rate = MOCK_SAMPLE_RATE

    def _setup_elevenlabs(self) -> bool:
        """Only ever called when OPENCLAW_TTS_BACKEND=elevenlabs was set explicitly."""
        elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY")
        if not elevenlabs_key:
            logger.info("ELEVENLABS_API_KEY not set; ElevenLabs unavailable")
            return False
        try:
            from elevenlabs import ElevenLabs
            self._elevenlabs_client = ElevenLabs(api_key=elevenlabs_key)
            self._backend = "elevenlabs"
            self.sample_rate = ELEVENLABS_SAMPLE_RATE
            logger.info("ElevenLabs TTS ready (premium, opt-in)")
            return True
        except ImportError:
            logger.warning("elevenlabs SDK not installed; not auto-installing, falling back")
            return False
        except Exception as e:
            logger.warning(f"ElevenLabs setup failed: {e}")
            return False

    def _setup_piper(self) -> bool:
        piper_bin = shutil.which(self._piper_bin)
        if not piper_bin and os.path.isfile(self._piper_bin):
            piper_bin = self._piper_bin
        if not piper_bin:
            logger.warning(f"piper binary not found ({self._piper_bin!r}); falling back")
            return False

        model_path, config_path = self._resolve_voice_paths()
        if model_path is None:
            logger.warning(
                f"Piper voice '{self._piper_voice_name}' not found under "
                f"{DEFAULT_PIPER_MODEL_DIR}; run "
                "`python scripts/download_models.py piper` to fetch it. Falling back."
            )
            return False

        try:
            with open(config_path) as f:
                config = json.load(f)
            sample_rate = int(config.get("audio", {}).get("sample_rate", 22050))
        except Exception as e:
            logger.warning(f"Could not read piper voice config {config_path}: {e}")
            return False

        self._piper_bin = piper_bin
        self._piper_model_path = model_path
        self._piper_config_path = config_path
        self.sample_rate = sample_rate
        self._backend = "piper"
        logger.info(f"Piper TTS ready (voice={self._piper_voice_name}, sample_rate={sample_rate})")
        return True

    def _resolve_voice_paths(self):
        """Resolve the configured voice name/path to (model.onnx, model.onnx.json)."""
        name = self._piper_voice_name
        candidate = Path(name)
        if candidate.suffix == ".onnx":
            model_path = candidate
        else:
            model_path = DEFAULT_PIPER_MODEL_DIR / f"{name}.onnx"
        config_path = Path(str(model_path) + ".json")
        if model_path.exists() and config_path.exists():
            return model_path, config_path
        return None, None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def synthesize(self, text: str) -> np.ndarray:
        """Non-streaming synthesis. Returns float32 samples in [-1, 1] (in-process only)."""
        chunks = [chunk async for chunk in self.synthesize_stream(text)]
        if not chunks:
            return np.zeros(1, dtype=np.float32)
        audio_bytes = b"".join(chunks)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        return audio_int16.astype(np.float32) / 32768.0

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized speech.

        Yields:
            Raw int16 PCM ("s16le") chunks at self.sample_rate. Never float32.
        """
        if not text or not text.strip():
            return

        if self._backend == "piper":
            async for chunk in self._piper_stream(text):
                yield chunk
        elif self._backend == "elevenlabs":
            async for chunk in self._elevenlabs_stream(text):
                yield chunk
        else:
            yield self._mock_audio_bytes(text)

    # ------------------------------------------------------------------ #
    # Piper (local, default)
    # ------------------------------------------------------------------ #

    async def _piper_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Run `piper --output-raw` as a subprocess and stream stdout as it arrives."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._piper_bin,
                "-m", str(self._piper_model_path),
                "-c", str(self._piper_config_path),
                "--output-raw",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            logger.error(f"Failed to launch piper: {e}")
            return

        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(text.encode("utf-8") + b"\n")
            await proc.stdin.drain()
        except Exception as e:
            logger.error(f"Failed writing to piper stdin: {e}")
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

        try:
            while True:
                chunk = await proc.stdout.read(_STDOUT_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            # Cancellation (barge-in / manual interrupt) lands here with
            # piper still running — kill it FIRST, otherwise awaiting its
            # exit blocks the interrupt path behind unread synthesis.
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            stderr = b""
            if proc.stderr is not None:
                try:
                    stderr = await proc.stderr.read()
                except Exception:
                    pass
            try:
                returncode = await proc.wait()
            except Exception:
                returncode = -1
            if returncode not in (0, -9):  # -9 = our own kill
                logger.error(f"piper exited {returncode}: {stderr.decode(errors='ignore')}")

    # ------------------------------------------------------------------ #
    # ElevenLabs (opt-in premium)
    # ------------------------------------------------------------------ #

    async def _elevenlabs_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream ElevenLabs audio without blocking the event loop.

        The SDK's text_to_speech.convert() is a synchronous generator, so it
        runs on a worker thread; chunks are relayed back through an
        asyncio.Queue as they're produced (not collected-then-replayed), so
        we keep the streaming latency benefit.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def producer():
            try:
                audio_generator = self._elevenlabs_client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    text=text,
                    model_id="eleven_turbo_v2_5",
                    output_format="pcm_24000",
                )
                for chunk in audio_generator:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        producer_future = loop.run_in_executor(None, producer)
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    logger.error(f"ElevenLabs streaming error: {item}")
                    break
                yield item
        finally:
            await producer_future

    # ------------------------------------------------------------------ #
    # Mock (silence fallback)
    # ------------------------------------------------------------------ #

    def _mock_audio_bytes(self, text: str) -> bytes:
        logger.debug(f"Mock TTS: {text[:50]!r}")
        n_samples = int(self.sample_rate * 0.5)  # 0.5s of silence
        return np.zeros(n_samples, dtype=np.int16).tobytes()


# main.py imports `ChatterboxTTS` — keep that name working unchanged.
ChatterboxTTS = TTSEngine

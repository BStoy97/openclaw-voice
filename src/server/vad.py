"""
Voice Activity Detection module.

Modern Silero VAD is stateful and requires exactly 512-sample frames at
16 kHz per call.  This wrapper owns the framing: arbitrary-size chunks go
in, one speech probability per complete 512-sample frame comes out.
"""

import os
from typing import List, Optional

import numpy as np
from loguru import logger

SAMPLE_RATE = 16000
FRAME_SIZE = 512  # samples per Silero frame @ 16 kHz
FRAME_SECS = FRAME_SIZE / SAMPLE_RATE  # 0.032 s


class VoiceActivityDetector:
    """Streaming Silero VAD with internal 512-sample framing.

    Backend ladder: silero_vad pip package (local, no network) →
    torch.hub (network) → energy gate (explicit override only) →
    fail-open (no model → probability 1.0).
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = SAMPLE_RATE):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model = None
        self._backend = "none"
        self._pending = np.zeros(0, dtype=np.float32)

        backend_pref = os.getenv("OPENCLAW_VAD_BACKEND", "silero").lower()
        if backend_pref == "energy":
            # Deterministic dev/test override: RMS gate instead of a model.
            self._backend = "energy"
            logger.info("VAD backend: energy (OPENCLAW_VAD_BACKEND override)")
        elif backend_pref in ("none", "off", "disabled"):
            logger.warning("VAD disabled via OPENCLAW_VAD_BACKEND; failing open")
        else:
            self._load_model()

    def _load_model(self):
        """Load Silero VAD: pip package first (no network), then torch.hub."""
        try:
            from silero_vad import load_silero_vad

            self.model = load_silero_vad()
            self._backend = "silero"
            logger.info("✅ Silero VAD loaded (silero_vad package)")
            return
        except Exception as e:
            logger.warning(f"silero_vad package unavailable: {e}")

        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
            self.model = model
            self._backend = "silero"
            logger.info("✅ Silero VAD loaded (torch.hub)")
        except Exception as e:
            logger.warning(f"VAD not available (fail-open): {e}")
            self.model = None
            self._backend = "none"

    def process(self, audio: np.ndarray) -> List[float]:
        """Feed an arbitrary-size chunk of 16 kHz float32 mono audio.

        Returns one speech probability per complete 512-sample frame
        consumed.  Leftover samples are buffered for the next call.
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if self._pending.size:
            audio = np.concatenate([self._pending, audio])
        probs: List[float] = []
        offset = 0
        while offset + FRAME_SIZE <= len(audio):
            probs.append(self._score_frame(audio[offset : offset + FRAME_SIZE]))
            offset += FRAME_SIZE
        self._pending = audio[offset:].copy()
        return probs

    def _score_frame(self, frame: np.ndarray) -> float:
        """Speech probability for exactly one 512-sample frame."""
        if self._backend == "energy":
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            return 1.0 if rms > 0.01 else 0.0
        if self.model is None:
            return 1.0  # fail open: assume speech
        try:
            import torch

            with torch.no_grad():
                return float(
                    self.model(torch.from_numpy(frame), self.sample_rate).item()
                )
        except Exception as e:
            logger.error(f"VAD error (fail-open): {e}")
            return 1.0

    def is_speech(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
        """Compat wrapper: True if mean frame probability exceeds threshold."""
        probs = self.process(audio)
        if not probs:
            # No complete frame yet: fail open only when no model is loaded.
            return self.model is None and self._backend != "energy"
        return float(np.mean(probs)) > self.threshold

    def reset(self):
        """Reset model state and the internal frame buffer."""
        self._pending = np.zeros(0, dtype=np.float32)
        if self.model is not None and hasattr(self.model, "reset_states"):
            try:
                self.model.reset_states()
            except Exception as e:
                logger.warning(f"VAD reset failed: {e}")

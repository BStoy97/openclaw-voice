"""
Server-driven turn-detection engine.

State machine (per WebSocket connection):

    IDLE ──sustained speech──▶ USER_SPEAKING ──min_silence──▶ PENDING_EOT
      ▲                            ▲    ▲                        │
      │                            │    └──speech resumes────────┤
      │                            │                             │
      └──────commit (semantic / ceiling / timeout / manual)──────┘

    AGENT_RESPONDING: entered via set_agent_responding(True) while the
    response pipeline (thinking + speaking) runs.  User speech there is
    evaluated with a lower VAD threshold (barge_responding_threshold —
    in-car echo cancellation attenuates double-talk) and accumulated into
    a rolling barge buffer (~1 s preroll + the speech run, capped at a
    5 s tail).  After barge_candidate_secs of accumulated speech the
    engine emits BARGE_CANDIDATE carrying the buffered audio WITHOUT
    changing state; the caller transcribes it and answers via
    resolve_barge():

        'cancel'   — stop phrase heard: clear the buffer, suppress further
                     candidates, stay AGENT_RESPONDING until
                     set_agent_responding(False)
        'takeover' — real talk-over: transition to USER_SPEAKING seeded
                     with the barge buffer as the new turn's frames
        'ignore'   — noise/filler: clear run + buffer, stay put

    Pure-VAD fallback: if a speech run reaches 1.2 s without any candidate
    being resolved (transcription unavailable), the engine emits legacy
    BARGE_IN and drops straight into USER_SPEAKING as before.

Semantic end-of-turn uses pipecat-ai's smart-turn-v3 ONNX model
(BSD-2-Clause).  Preprocessing/inference adapted with attribution from
https://github.com/pipecat-ai/smart-turn (inference.py, audio_utils.py).
"""

import dataclasses
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from .vad import FRAME_SECS, FRAME_SIZE, SAMPLE_RATE, VoiceActivityDetector

# --------------------------------------------------------------------------- #
# Events / states
# --------------------------------------------------------------------------- #

USER_SPEECH_STARTED = "user_speech_started"
EOT_PENDING = "eot_pending"
TURN_COMMITTED = "turn_committed"
BARGE_IN = "barge_in"
BARGE_CANDIDATE = "barge_candidate"

# TURN_COMMITTED reasons
REASON_SEMANTIC = "semantic"   # smart-turn gate said the utterance is complete
REASON_CEILING = "ceiling"     # patience ceiling hit
REASON_MANUAL = "manual"       # force_commit() (legacy stop_listening)
REASON_TIMEOUT = "timeout"     # semantic gate unavailable: 2x min_silence fallback


class TurnState(str, Enum):
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    PENDING_EOT = "pending_eot"
    AGENT_RESPONDING = "agent_responding"


@dataclass
class TurnEvent:
    type: str
    reason: Optional[str] = None
    audio: Optional[np.ndarray] = None
    silence_secs: Optional[float] = None  # user-silence at commit (perceived-gap math)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

ENV_PREFIX = "OPENCLAW_TURN_"
_MAX_PATIENCE_SECS = 20.0
_TRUTHY = {"1", "true", "yes", "on"}


@dataclass
class TurnConfig:
    """Turn-detection tuning. Every field is readable from the environment
    with prefix ``OPENCLAW_TURN_`` (e.g. ``OPENCLAW_TURN_MIN_SILENCE_SECS``)
    and overridable per session via the ``session_start`` config payload."""

    start_threshold: float = 0.5          # VAD prob to count a frame as speech
    min_speech_frames: int = 8            # frames (~0.25 s) above threshold to start a turn
    min_silence_secs: float = 1.8         # silence before the first end-of-turn check
    recheck_interval_secs: float = 2.0    # semantic re-check cadence while pending
    patience_ceiling_secs: float = 18.0   # absolute max silence before forced commit
    semantic_enabled: bool = True         # use the smart-turn gate
    barge_in_min_speech_secs: float = 0.5 # sustained speech to interrupt the agent
    semantic_threshold: float = 0.5       # P(turn complete) needed to commit
    barge_candidate_secs: float = 0.15    # accumulated speech (AGENT_RESPONDING) to emit BARGE_CANDIDATE
    barge_gap_frames: int = 8             # non-speech frames tolerated inside a barge run (~256 ms)
    barge_responding_threshold: float = 0.35  # VAD prob counted as speech during AGENT_RESPONDING (AEC attenuates)
    stop_phrases: str = (                 # comma-separated; parsed to stop_phrase_list
        "stop,hold on,wait,be quiet,quiet,shut up,that's enough,enough,"
        "okay stop,ok stop,pause,never mind,nevermind,interrupt,"
        "interruption,stop talking,hush"
    )

    def __post_init__(self):
        self.clamp()

    def clamp(self):
        """Coerce + clamp all fields to safe ranges (ceiling hard-capped at 20 s)."""
        self.start_threshold = min(max(float(self.start_threshold), 0.0), 1.0)
        self.min_speech_frames = max(1, int(self.min_speech_frames))
        self.min_silence_secs = max(0.2, float(self.min_silence_secs))
        self.recheck_interval_secs = max(0.2, float(self.recheck_interval_secs))
        self.patience_ceiling_secs = min(
            max(float(self.patience_ceiling_secs), self.min_silence_secs),
            _MAX_PATIENCE_SECS,
        )
        self.semantic_enabled = (
            self.semantic_enabled
            if isinstance(self.semantic_enabled, bool)
            else str(self.semantic_enabled).strip().lower() in _TRUTHY
        )
        self.barge_in_min_speech_secs = max(0.1, float(self.barge_in_min_speech_secs))
        self.semantic_threshold = min(max(float(self.semantic_threshold), 0.0), 1.0)
        self.barge_candidate_secs = max(0.1, float(self.barge_candidate_secs))
        self.barge_gap_frames = max(1, int(self.barge_gap_frames))
        self.barge_responding_threshold = min(
            max(float(self.barge_responding_threshold), 0.0), 1.0
        )
        self.stop_phrases = str(self.stop_phrases)
        # Parsed once here; per-session overrides flow through from_env ->
        # _parse (str passthrough) -> this clamp, so they re-parse too.
        self.stop_phrase_list: List[str] = [
            p.strip().lower() for p in self.stop_phrases.split(",") if p.strip()
        ]

    @classmethod
    def from_env(cls, overrides: Optional[dict] = None) -> "TurnConfig":
        """Build a config from OPENCLAW_TURN_* env vars, then apply optional
        per-session overrides (unknown keys and bad values are ignored)."""
        kwargs = {}
        for f in dataclasses.fields(cls):
            raw = os.getenv(ENV_PREFIX + f.name.upper())
            if raw is None:
                continue
            parsed = cls._parse(f, raw)
            if parsed is not None:
                kwargs[f.name] = parsed
        if overrides:
            valid = {f.name: f for f in dataclasses.fields(cls)}
            for key, value in overrides.items():
                f = valid.get(key)
                if f is None:
                    logger.warning(f"Ignoring unknown turn config key: {key!r}")
                    continue
                parsed = cls._parse(f, value)
                if parsed is not None:
                    kwargs[key] = parsed
        return cls(**kwargs)

    @staticmethod
    def _parse(field: dataclasses.Field, value):
        try:
            if field.type in (bool, "bool"):
                if isinstance(value, bool):
                    return value
                return str(value).strip().lower() in _TRUTHY
            if field.type in (int, "int"):
                return int(value)
            if field.type in (str, "str"):
                return str(value)
            return float(value)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring invalid turn config value {field.name}={value!r}")
            return None


# --------------------------------------------------------------------------- #
# Semantic end-of-turn gate (smart-turn-v3)
# --------------------------------------------------------------------------- #

DEFAULT_SMART_TURN_DIR = (
    Path(__file__).resolve().parent.parent.parent / "models" / "smart-turn"
)


class SmartTurnGate:
    """Semantic end-of-turn classifier.

    Wraps pipecat-ai/smart-turn-v3 (whisper-tiny encoder + linear head,
    ONNX, BSD-2-Clause).  Input: 16 kHz mono float32, last 8 s of the turn
    (leading-padded with zeros).  Output: sigmoid P(turn complete).

    Inference code adapted from pipecat-ai/smart-turn inference.py
    (BSD-2-Clause, Copyright Daily).
    """

    WINDOW_SECS = 8

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        self.threshold = threshold
        self.available = False
        self._session = None
        self._feature_extractor = None
        try:
            path = Path(model_path) if model_path else self._find_model()
            if path is None or not path.exists():
                raise FileNotFoundError(
                    f"No smart-turn .onnx model found under {DEFAULT_SMART_TURN_DIR} "
                    "(run: python scripts/download_models.py smart-turn)"
                )
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(path), sess_options=so)

            from transformers import WhisperFeatureExtractor

            self._feature_extractor = WhisperFeatureExtractor(
                chunk_length=self.WINDOW_SECS
            )
            self.available = True
            logger.info(f"✅ Smart-turn gate ready ({path.name})")
        except Exception as e:
            logger.warning(f"Smart-turn gate unavailable: {e}")

    @staticmethod
    def _find_model() -> Optional[Path]:
        matches = sorted(DEFAULT_SMART_TURN_DIR.glob("*.onnx"))
        return matches[-1] if matches else None

    def predict(self, audio: np.ndarray) -> Optional[float]:
        """Return P(turn complete) in [0, 1] for the given audio, or None on
        failure (caller should degrade to non-semantic behavior)."""
        if not self.available:
            return None
        try:
            max_samples = self.WINDOW_SECS * SAMPLE_RATE
            audio = np.asarray(audio, dtype=np.float32)
            if len(audio) > max_samples:
                audio = audio[-max_samples:]  # keep the end of the turn
            elif len(audio) < max_samples:
                audio = np.pad(audio, (max_samples - len(audio), 0))
            inputs = self._feature_extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
                padding="max_length",
                max_length=max_samples,
                truncation=True,
                do_normalize=True,
            )
            feats = inputs.input_features.astype(np.float32)  # (1, 80, 800)
            outputs = self._session.run(None, {"input_features": feats})
            return float(outputs[0][0].item())  # sigmoid already applied in-graph
        except Exception as e:
            logger.error(f"Smart-turn inference failed: {e}")
            return None


# --------------------------------------------------------------------------- #
# Turn engine
# --------------------------------------------------------------------------- #

class TurnEngine:
    """Per-connection turn-detection state machine.

    Synchronous: ``feed()`` is called from the WebSocket receive loop with
    each incoming audio chunk (16 kHz float32 mono) and the current
    monotonic time, and returns zero or more TurnEvents.  Timing advances
    only when audio arrives (continuous-mode clients stream constantly).
    """

    _PREROLL_SECS = 1.0            # audio kept before speech onset
    _RESUME_SPEECH_FRAMES = 2      # speech frames to cancel PENDING_EOT
    _BARGE_FALLBACK_SECS = 1.2     # unresolved speech run before legacy BARGE_IN
    _BARGE_BUFFER_MAX_SECS = 5.0   # rolling barge buffer tail cap
    _NOISE_EMA_ALPHA = 0.05
    _NOISE_PROB_CEILING = 0.3      # frames below this VAD prob feed the noise EMA

    def __init__(
        self,
        config: Optional[TurnConfig] = None,
        vad: Optional[VoiceActivityDetector] = None,
        gate: Optional[SmartTurnGate] = None,
    ):
        self.config = config or TurnConfig.from_env()
        self.vad = vad if vad is not None else VoiceActivityDetector(
            threshold=self.config.start_threshold
        )
        if gate is not None:
            self.gate = gate
        elif self.config.semantic_enabled:
            self.gate = SmartTurnGate(threshold=self.config.semantic_threshold)
        else:
            self.gate = None
        if self.gate is not None and not getattr(self.gate, "available", True):
            self.gate = None
        self.semantic_active = bool(self.config.semantic_enabled and self.gate)

        self.state = TurnState.IDLE
        self.agent_responding = False
        self.noise_floor_rms: Optional[float] = None  # diagnostic EMA, never gates
        self.last_semantic_prob: Optional[float] = None

        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll: deque = deque(maxlen=max(1, int(self._PREROLL_SECS / FRAME_SECS)))
        self._recent_speech: deque = deque(
            maxlen=max(self.config.min_speech_frames * 2, 16)
        )
        self._turn_frames: List[np.ndarray] = []
        self._last_speech_time = 0.0
        self._next_semantic_check = 0.0
        self._resume_speech_run = 0
        self._barge_speech_run = 0
        self._barge_nonspeech_run = 0
        self._barge_fallback_run = 0
        self._barge_candidates = 0
        self._barge_buffer: List[np.ndarray] = []
        self._barge_suppressed = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def feed(self, audio_chunk: np.ndarray, now: float) -> List[TurnEvent]:
        """Process an audio chunk; advance the state machine; return events."""
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        if self._pending.size:
            audio_chunk = np.concatenate([self._pending, audio_chunk])
        events: List[TurnEvent] = []
        offset = 0
        while offset + FRAME_SIZE <= len(audio_chunk):
            frame = audio_chunk[offset : offset + FRAME_SIZE].copy()
            offset += FRAME_SIZE
            probs = self.vad.process(frame)
            prob = probs[0] if probs else 1.0
            events.extend(self._process_frame(frame, prob, now))
        self._pending = audio_chunk[offset:].copy()
        return events

    def set_agent_responding(self, active: bool) -> None:
        """Signal that the agent response pipeline (thinking/speaking) is
        active, so incoming speech is evaluated for barge-in."""
        self.agent_responding = active
        self._barge_suppressed = False
        if active:
            self.state = TurnState.AGENT_RESPONDING
            self._turn_frames = []
            self._recent_speech.clear()
            self._reset_barge()
        elif self.state == TurnState.AGENT_RESPONDING:
            # Barge-in already moved us to USER_SPEAKING; don't clobber that.
            # NOTE: the barge buffer is deliberately NOT cleared here — a
            # pending BARGE_CANDIDATE may still be resolved as 'takeover'
            # after the response task's cleanup already flipped this flag.
            self.state = TurnState.IDLE
            self._recent_speech.clear()

    def resolve_barge(self, decision: str, now: float) -> None:
        """Answer an emitted BARGE_CANDIDATE (transcribe-then-decide).

        'cancel'   — stop phrase: clear buffer, suppress further candidates;
                     state is left alone (caller follows up with
                     set_agent_responding(False)).
        'takeover' — genuine talk-over: become USER_SPEAKING with the barge
                     buffer seeded as the new turn's frames. Emits nothing —
                     the caller already knows and does its own bookkeeping.
        'ignore'   — noise/filler: clear run + buffer, keep listening for
                     the next barge attempt.
        """
        if decision == "takeover":
            buffer = self._barge_buffer
            self._reset_barge()
            self._start_turn(now)
            if buffer:
                self._turn_frames = buffer  # includes ~1 s preroll
        elif decision == "cancel":
            self._reset_barge()
            self._barge_suppressed = True
        elif decision == "ignore":
            self._reset_barge()
        else:
            raise ValueError(f"Unknown barge decision: {decision!r}")

    def force_commit(self) -> List[TurnEvent]:
        """Commit whatever turn audio has accumulated (legacy stop_listening)."""
        if self.state in (TurnState.USER_SPEAKING, TurnState.PENDING_EOT) and self._turn_frames:
            return [self._commit(REASON_MANUAL)]
        return []

    def reset(self) -> None:
        """Full reset: state, buffers, VAD model state."""
        self.state = TurnState.IDLE
        self.agent_responding = False
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._recent_speech.clear()
        self._turn_frames = []
        self._last_speech_time = 0.0
        self._next_semantic_check = 0.0
        self._resume_speech_run = 0
        self._reset_barge()
        self._barge_suppressed = False
        self.last_semantic_prob = None
        if hasattr(self.vad, "reset"):
            self.vad.reset()

    def diagnostics(self) -> dict:
        return {
            "state": self.state.value,
            "semantic_active": self.semantic_active,
            "noise_floor_rms": self.noise_floor_rms,
            "last_semantic_prob": self.last_semantic_prob,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _process_frame(
        self, frame: np.ndarray, prob: float, now: float
    ) -> List[TurnEvent]:
        events: List[TurnEvent] = []
        is_speech = prob >= self.config.start_threshold

        # Adaptive noise floor: EMA of RMS over frames VAD scores as non-speech.
        # Diagnostic only — VAD probability remains the primary gate.
        if prob < self._NOISE_PROB_CEILING:
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            if self.noise_floor_rms is None:
                self.noise_floor_rms = rms
            else:
                a = self._NOISE_EMA_ALPHA
                self.noise_floor_rms = (1 - a) * self.noise_floor_rms + a * rms

        self._preroll.append(frame)

        if self.state == TurnState.AGENT_RESPONDING:
            events.extend(self._handle_barge_frame(frame, prob, now))
        elif self.state == TurnState.IDLE:
            self._recent_speech.append(is_speech)
            if sum(self._recent_speech) >= self.config.min_speech_frames:
                self._start_turn(now)
                events.append(TurnEvent(USER_SPEECH_STARTED))
        elif self.state == TurnState.USER_SPEAKING:
            self._turn_frames.append(frame)
            if is_speech:
                self._last_speech_time = now
            else:
                silence = now - self._last_speech_time
                if self.semantic_active:
                    if silence >= self.config.min_silence_secs:
                        self.state = TurnState.PENDING_EOT
                        self._resume_speech_run = 0
                        events.append(TurnEvent(EOT_PENDING))
                        events.extend(self._semantic_check(now))
                else:
                    # Gate unavailable: legacy-ish fixed silence commit.
                    if silence >= self.config.min_silence_secs * 2:
                        events.append(self._commit(REASON_TIMEOUT, now))
        elif self.state == TurnState.PENDING_EOT:
            self._turn_frames.append(frame)
            if is_speech:
                self._resume_speech_run += 1
                if self._resume_speech_run >= self._RESUME_SPEECH_FRAMES:
                    # User resumed speaking: cancel the pending end-of-turn.
                    self.state = TurnState.USER_SPEAKING
                    self._last_speech_time = now
                    self._resume_speech_run = 0
            else:
                self._resume_speech_run = 0
                silence = now - self._last_speech_time
                if silence >= self.config.patience_ceiling_secs:
                    events.append(self._commit(REASON_CEILING, now))
                elif now >= self._next_semantic_check:
                    events.extend(self._semantic_check(now))

        return events

    def _handle_barge_frame(
        self, frame: np.ndarray, prob: float, now: float
    ) -> List[TurnEvent]:
        """Barge detection while the agent is speaking.

        Speech frames (>= barge_responding_threshold; lower than
        start_threshold because in-car AEC attenuates double-talk) grow a
        rolling buffer seeded from the ~1 s preroll.  Every
        barge_candidate_secs of accumulated speech emits BARGE_CANDIDATE
        with the buffered audio for the caller to transcribe and resolve;
        an unresolved run reaching _BARGE_FALLBACK_SECS falls back to the
        legacy pure-VAD BARGE_IN.
        """
        if self._barge_suppressed:
            return []
        is_speech = prob >= self.config.barge_responding_threshold
        if is_speech:
            if not self._barge_buffer:
                # Preroll already contains the current frame (appended in
                # _process_frame before dispatch).
                self._barge_buffer = list(self._preroll)
            else:
                self._barge_buffer.append(frame)
            self._barge_speech_run += 1
            self._barge_fallback_run += 1
            self._barge_nonspeech_run = 0
        else:
            if self._barge_buffer:
                self._barge_buffer.append(frame)
            self._barge_nonspeech_run += 1
            if self._barge_nonspeech_run >= self.config.barge_gap_frames:
                # Run died. Keep the buffer only if a candidate is already
                # in flight (it keeps growing across candidates within one
                # agent response); otherwise start fresh next burst.
                self._barge_speech_run = 0
                self._barge_fallback_run = 0
                if self._barge_candidates == 0:
                    self._barge_buffer = []
        # Cap the buffer to a trailing window (frames are FRAME_SIZE each).
        max_frames = max(1, int(self._BARGE_BUFFER_MAX_SECS / FRAME_SECS))
        if len(self._barge_buffer) > max_frames:
            del self._barge_buffer[: len(self._barge_buffer) - max_frames]

        fallback_frames = max(1, int(round(self._BARGE_FALLBACK_SECS / FRAME_SECS)))
        if self._barge_fallback_run >= fallback_frames:
            # Legacy pure-VAD fallback: no one resolved our candidates
            # (transcription unavailable) — barge in directly.
            buffer = self._barge_buffer
            self._reset_barge()
            self._start_turn(now)
            if buffer:
                self._turn_frames = buffer
            return [TurnEvent(BARGE_IN), TurnEvent(USER_SPEECH_STARTED)]

        candidate_frames = max(
            1, int(round(self.config.barge_candidate_secs / FRAME_SECS))
        )
        if self._barge_speech_run >= candidate_frames:
            self._barge_speech_run = 0  # re-arm: repeated attempts re-emit
            self._barge_candidates += 1
            return [
                TurnEvent(BARGE_CANDIDATE, audio=np.concatenate(self._barge_buffer))
            ]
        return []

    def _reset_barge(self) -> None:
        self._barge_speech_run = 0
        self._barge_nonspeech_run = 0
        self._barge_fallback_run = 0
        self._barge_candidates = 0
        self._barge_buffer = []

    def _start_turn(self, now: float) -> None:
        self.state = TurnState.USER_SPEAKING
        self._turn_frames = list(self._preroll)  # include onset pre-roll
        self._last_speech_time = now
        self._recent_speech.clear()
        self._resume_speech_run = 0

    def _semantic_check(self, now: float) -> List[TurnEvent]:
        self._next_semantic_check = now + self.config.recheck_interval_secs
        prob = self.gate.predict(self._turn_audio()) if self.gate else None
        if prob is None:
            # Gate broke at runtime: degrade to the non-semantic path.
            logger.warning("Semantic gate failed; degrading to silence-timeout mode")
            self.semantic_active = False
            self.state = TurnState.USER_SPEAKING
            return []
        self.last_semantic_prob = prob
        if prob > self.config.semantic_threshold:
            return [self._commit(REASON_SEMANTIC, now)]
        return []

    def _commit(self, reason: str, now: Optional[float] = None) -> TurnEvent:
        audio = self._turn_audio()
        self.state = TurnState.IDLE
        self._turn_frames = []
        self._recent_speech.clear()
        self._resume_speech_run = 0
        silence = None
        if now is not None and self._last_speech_time:
            silence = max(0.0, now - self._last_speech_time)
        return TurnEvent(TURN_COMMITTED, reason=reason, audio=audio, silence_secs=silence)

    def _turn_audio(self) -> np.ndarray:
        if not self._turn_frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._turn_frames)

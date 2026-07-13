"""
Unit tests for the turn-detection engine (src/server/turn.py).

The state machine is driven with synthetic audio (zeros for silence, noise
bursts for "speech") plus a scripted VAD (injected probability sequences)
and a mocked semantic gate, so no models are needed.  One integration test
exercises the real smart-turn ONNX model if it has been downloaded.
"""

import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.server.turn import (  # noqa: E402
    BARGE_CANDIDATE,
    BARGE_IN,
    EOT_PENDING,
    REASON_CEILING,
    REASON_MANUAL,
    REASON_SEMANTIC,
    REASON_TIMEOUT,
    TURN_COMMITTED,
    USER_SPEECH_STARTED,
    SmartTurnGate,
    TurnConfig,
    TurnEngine,
    TurnState,
)
from src.server.vad import FRAME_SECS, FRAME_SIZE  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "smart-turn"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class ScriptedVAD:
    """VAD stub: returns pre-scripted probabilities, one per 512-sample frame."""

    def __init__(self):
        self.queue = deque()

    def push(self, probs):
        self.queue.extend(probs)

    def process(self, audio):
        n_frames = len(audio) // FRAME_SIZE
        return [self.queue.popleft() if self.queue else 0.0 for _ in range(n_frames)]

    def reset(self):
        self.queue.clear()


class FakeGate:
    """Semantic gate stub: returns scripted P(complete) values."""

    available = True

    def __init__(self, probs):
        # probs: scalar (returned forever) or list (popped; last value sticks)
        self._probs = list(probs) if isinstance(probs, (list, tuple)) else [probs]
        self.calls = 0

    def predict(self, audio):
        self.calls += 1
        if len(self._probs) > 1:
            return self._probs.pop(0)
        return self._probs[0]


def make_engine(gate, **config_kwargs):
    config = TurnConfig(**config_kwargs)
    return TurnEngine(config=config, vad=ScriptedVAD(), gate=gate)


def drive(engine, probs, start_time, speech_amplitude=0.1):
    """Feed one 512-sample frame per scripted probability; collect events."""
    events = []
    t = start_time
    for p in probs:
        frame = (
            (np.random.default_rng(0).standard_normal(FRAME_SIZE) * speech_amplitude)
            if p >= 0.5 else np.zeros(FRAME_SIZE)
        ).astype(np.float32)
        engine.vad.push([p])
        events.extend(engine.feed(frame, t))
        t += FRAME_SECS
    return events, t


def frames_for(secs):
    return int(secs / FRAME_SECS) + 2  # small margin past the boundary


# --------------------------------------------------------------------------- #
# State machine tests
# --------------------------------------------------------------------------- #

class TestSemanticCommit:
    def test_speech_then_silence_commits_semantic(self):
        """(a) speech → 1.8 s silence + gate says complete → semantic commit."""
        gate = FakeGate(0.9)
        engine = make_engine(gate)

        events, t = drive(engine, [0.9] * 10, 0.0)
        assert [e.type for e in events] == [USER_SPEECH_STARTED]
        assert engine.state == TurnState.USER_SPEAKING

        events, _ = drive(engine, [0.0] * frames_for(1.8), t)
        types = [e.type for e in events]
        assert EOT_PENDING in types
        assert TURN_COMMITTED in types
        committed = next(e for e in events if e.type == TURN_COMMITTED)
        assert committed.reason == REASON_SEMANTIC
        assert isinstance(committed.audio, np.ndarray)
        assert len(committed.audio) > 0
        assert committed.audio.dtype == np.float32
        assert engine.state == TurnState.IDLE
        assert gate.calls == 1

    def test_no_commit_before_min_silence(self):
        gate = FakeGate(0.9)
        engine = make_engine(gate)
        events, t = drive(engine, [0.9] * 10, 0.0)
        # only 1 second of silence: not enough for the first EOT check
        events, _ = drive(engine, [0.0] * frames_for(1.0), t)
        assert all(e.type not in (EOT_PENDING, TURN_COMMITTED) for e in events)
        assert gate.calls == 0


class TestCeiling:
    def test_incomplete_stays_pending_until_ceiling(self):
        """(b) gate says incomplete → stays pending, commits at the ceiling."""
        gate = FakeGate(0.1)
        engine = make_engine(gate)

        _, t = drive(engine, [0.9] * 10, 0.0)
        events, _ = drive(engine, [0.0] * frames_for(18.5), t)

        committed = [e for e in events if e.type == TURN_COMMITTED]
        assert len(committed) == 1
        assert committed[0].reason == REASON_CEILING
        # gate re-checked during the wait (every recheck_interval_secs=2.0)
        assert gate.calls >= 5
        assert engine.state == TurnState.IDLE


class TestResumedSpeech:
    def test_resumed_speech_cancels_pending(self):
        """(c) speech resuming during PENDING_EOT returns to USER_SPEAKING."""
        gate = FakeGate([0.1, 0.1, 0.9])
        engine = make_engine(gate)

        _, t = drive(engine, [0.9] * 10, 0.0)
        events, t = drive(engine, [0.0] * frames_for(2.0), t)
        assert engine.state == TurnState.PENDING_EOT
        assert not [e for e in events if e.type == TURN_COMMITTED]

        # user resumes speaking
        events, t = drive(engine, [0.9] * 10, t)
        assert engine.state == TurnState.USER_SPEAKING
        assert not [e for e in events if e.type == TURN_COMMITTED]
        # no duplicate turn_started for the same turn
        assert not [e for e in events if e.type == USER_SPEECH_STARTED]

        # now go silent again; gate eventually says complete
        events, _ = drive(engine, [0.0] * frames_for(6.0), t)
        committed = [e for e in events if e.type == TURN_COMMITTED]
        assert len(committed) == 1
        assert committed[0].reason == REASON_SEMANTIC
        # committed audio spans both speech segments plus silences
        assert len(committed[0].audio) > 10 * FRAME_SIZE * 2


class TestBargeIn:
    def test_short_blip_does_not_barge(self):
        """(d) a 0.2 s blip during agent response must not interrupt."""
        engine = make_engine(FakeGate(0.9))
        engine.set_agent_responding(True)
        assert engine.state == TurnState.AGENT_RESPONDING

        events, t = drive(engine, [0.9] * 6, 0.0)  # ~0.19 s
        assert not [e for e in events if e.type in (BARGE_IN, BARGE_CANDIDATE)]
        events, _ = drive(engine, [0.0] * 20, t)
        assert not [e for e in events if e.type in (BARGE_IN, BARGE_CANDIDATE)]
        assert engine.state == TurnState.AGENT_RESPONDING

    def test_sustained_speech_emits_candidate_not_state_change(self):
        """(d) >= barge_candidate_secs sustained speech during agent
        response emits BARGE_CANDIDATE with buffered audio, WITHOUT
        changing state (main.py transcribes and decides)."""
        engine = make_engine(FakeGate(0.9))
        engine.set_agent_responding(True)

        events, _ = drive(engine, [0.9] * frames_for(0.6), 0.0)
        candidates = [e for e in events if e.type == BARGE_CANDIDATE]
        assert len(candidates) == 1
        assert isinstance(candidates[0].audio, np.ndarray)
        assert len(candidates[0].audio) > 0
        assert not [e for e in events if e.type == BARGE_IN]
        assert not [e for e in events if e.type == USER_SPEECH_STARTED]
        assert engine.state == TurnState.AGENT_RESPONDING

    def test_unresolved_run_falls_back_to_legacy_barge_in(self):
        """A 1.2 s speech run with no resolve_barge() falls back to the
        pure-VAD BARGE_IN exactly as before (transcription unavailable)."""
        engine = make_engine(FakeGate(0.9))
        engine.set_agent_responding(True)

        events, _ = drive(engine, [0.9] * frames_for(1.3), 0.0)
        types = [e.type for e in events]
        assert BARGE_IN in types
        assert USER_SPEECH_STARTED in types
        # Candidates were emitted along the way (at 0.35 s cadence).
        assert types.count(BARGE_CANDIDATE) >= 2
        assert engine.state == TurnState.USER_SPEAKING
        # The barged turn is seeded with the buffered speech.
        committed = engine.force_commit()
        assert len(committed) == 1
        assert len(committed[0].audio) > 0

        # main.py cancels the response and clears the flag; the new turn
        # must survive that.
        engine.set_agent_responding(False)

    def test_no_barge_when_agent_idle(self):
        engine = make_engine(FakeGate(0.9))
        events, _ = drive(engine, [0.9] * frames_for(0.6), 0.0)
        assert not [e for e in events if e.type in (BARGE_IN, BARGE_CANDIDATE)]


class TestBargeCandidate:
    """Transcribe-then-decide barge-in: candidate emission + resolve paths."""

    def test_candidate_at_low_prob_with_gaps(self):
        """0.35 s of accumulated speech at prob 0.4 — below start_threshold
        (0.5) but above barge_responding_threshold (0.35) — with a gap up
        to barge_gap_frames still emits a candidate (in-car AEC ducking)."""
        engine = make_engine(FakeGate(0.9))
        assert engine.config.start_threshold == 0.5  # 0.4 would NOT start a turn
        engine.set_agent_responding(True)

        # 5 speech frames, 7-frame gap (< barge_gap_frames=8), 6 more speech
        probs = [0.4] * 5 + [0.0] * 7 + [0.4] * 6  # 11 speech frames ~0.35 s
        events, _ = drive(engine, probs, 0.0)
        candidates = [e for e in events if e.type == BARGE_CANDIDATE]
        assert len(candidates) == 1
        # Buffer spans the whole run (speech + gap frames)
        assert len(candidates[0].audio) >= 11 * FRAME_SIZE
        assert engine.state == TurnState.AGENT_RESPONDING

    def test_gap_beyond_tolerance_resets_run(self):
        engine = make_engine(FakeGate(0.9))
        engine.set_agent_responding(True)

        probs = [0.4] * 5 + [0.0] * 9 + [0.4] * 6  # gap 9 >= 8: run dies
        events, _ = drive(engine, probs, 0.0)
        assert not [e for e in events if e.type in (BARGE_CANDIDATE, BARGE_IN)]
        assert engine.state == TurnState.AGENT_RESPONDING

    def test_repeated_candidates_reemit_with_growing_buffer(self):
        """Each additional barge_candidate_secs of speech re-emits, and the
        buffer keeps growing across candidates within one response."""
        engine = make_engine(FakeGate(0.9))
        engine.set_agent_responding(True)

        events, _ = drive(engine, [0.4] * 23, 0.0)  # 2 x ~0.35 s runs
        candidates = [e for e in events if e.type == BARGE_CANDIDATE]
        assert len(candidates) == 2
        assert len(candidates[1].audio) > len(candidates[0].audio)

    def test_resolve_takeover_seeds_turn_with_buffer(self):
        engine = make_engine(FakeGate(0.9))
        engine.set_agent_responding(True)

        # Drive exactly to the candidate boundary so the buffer and the
        # candidate audio are identical when we resolve.
        n = max(1, int(round(engine.config.barge_candidate_secs / FRAME_SECS)))
        events, t = drive(engine, [0.4] * n, 0.0)
        candidate = next(e for e in events if e.type == BARGE_CANDIDATE)

        engine.resolve_barge("takeover", t)
        assert engine.state == TurnState.USER_SPEAKING
        # main.py clears the flag right after; the new turn must survive.
        engine.set_agent_responding(False)
        assert engine.state == TurnState.USER_SPEAKING

        committed = engine.force_commit()
        assert len(committed) == 1
        # Seeded exactly with the barge buffer (== the candidate audio,
        # since nothing was fed in between).
        assert len(committed[0].audio) == len(candidate.audio)

    def test_resolve_cancel_suppresses_until_agent_done(self):
        engine = make_engine(FakeGate(0.9))
        engine.set_agent_responding(True)

        events, t = drive(engine, [0.4] * 12, 0.0)
        assert [e for e in events if e.type == BARGE_CANDIDATE]

        engine.resolve_barge("cancel", t)
        assert engine.state == TurnState.AGENT_RESPONDING

        # Further speech while cancelled emits nothing (suppressed).
        events, t = drive(engine, [0.9] * 20, t)
        assert events == []

        engine.set_agent_responding(False)
        assert engine.state == TurnState.IDLE

        # Suppression is lifted: normal turn-taking works again.
        events, _ = drive(engine, [0.9] * 10, t)
        assert [e.type for e in events] == [USER_SPEECH_STARTED]

    def test_resolve_ignore_allows_new_candidate(self):
        engine = make_engine(FakeGate(0.9))
        engine.set_agent_responding(True)

        events, t = drive(engine, [0.4] * 12, 0.0)
        assert [e for e in events if e.type == BARGE_CANDIDATE]

        engine.resolve_barge("ignore", t)
        assert engine.state == TurnState.AGENT_RESPONDING

        # A fresh run re-emits a candidate.
        events, _ = drive(engine, [0.4] * 12, t)
        assert [e for e in events if e.type == BARGE_CANDIDATE]

    def test_resolve_unknown_decision_raises(self):
        engine = make_engine(FakeGate(0.9))
        with pytest.raises(ValueError):
            engine.resolve_barge("maybe", 0.0)


class TestConfig:
    def test_ceiling_clamped_to_20(self):
        """(e) patience ceiling is hard-capped at 20 s."""
        assert TurnConfig(patience_ceiling_secs=30).patience_ceiling_secs == 20.0
        assert TurnConfig(patience_ceiling_secs=18).patience_ceiling_secs == 18.0

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_TURN_PATIENCE_CEILING_SECS", "25")
        monkeypatch.setenv("OPENCLAW_TURN_MIN_SILENCE_SECS", "2.5")
        monkeypatch.setenv("OPENCLAW_TURN_SEMANTIC_ENABLED", "false")
        monkeypatch.setenv("OPENCLAW_TURN_MIN_SPEECH_FRAMES", "12")
        config = TurnConfig.from_env()
        assert config.patience_ceiling_secs == 20.0  # clamped
        assert config.min_silence_secs == 2.5
        assert config.semantic_enabled is False
        assert config.min_speech_frames == 12

    def test_overrides_validated_and_clamped(self):
        config = TurnConfig.from_env(overrides={
            "patience_ceiling_secs": 99,
            "min_silence_secs": "not-a-number",  # ignored
            "unknown_key": 1,                     # ignored
            "start_threshold": 0.7,
        })
        assert config.patience_ceiling_secs == 20.0
        assert config.min_silence_secs == 1.8  # default kept
        assert config.start_threshold == 0.7

    def test_barge_defaults(self):
        config = TurnConfig()
        assert config.barge_candidate_secs == 0.35
        assert config.barge_gap_frames == 8
        assert config.barge_responding_threshold == 0.35
        for phrase in ("stop", "hold on", "wait", "that's enough", "nevermind"):
            assert phrase in config.stop_phrase_list

    def test_barge_fields_clamped(self):
        assert TurnConfig(barge_candidate_secs=0.01).barge_candidate_secs == 0.1
        assert TurnConfig(barge_gap_frames=0).barge_gap_frames == 1
        assert TurnConfig(barge_responding_threshold=1.5).barge_responding_threshold == 1.0
        assert TurnConfig(barge_responding_threshold=-1).barge_responding_threshold == 0.0

    def test_barge_fields_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_TURN_BARGE_CANDIDATE_SECS", "0.5")
        monkeypatch.setenv("OPENCLAW_TURN_BARGE_GAP_FRAMES", "12")
        monkeypatch.setenv("OPENCLAW_TURN_BARGE_RESPONDING_THRESHOLD", "0.2")
        monkeypatch.setenv("OPENCLAW_TURN_STOP_PHRASES", "halt, Cease ,desist")
        config = TurnConfig.from_env()
        assert config.barge_candidate_secs == 0.5
        assert config.barge_gap_frames == 12
        assert config.barge_responding_threshold == 0.2
        assert config.stop_phrase_list == ["halt", "cease", "desist"]

    def test_stop_phrases_session_override_str_passthrough(self):
        """Per-session override of a str field flows through _parse."""
        config = TurnConfig.from_env(overrides={
            "stop_phrases": "Red Light, green light",
            "barge_responding_threshold": 0.25,
        })
        assert config.stop_phrase_list == ["red light", "green light"]
        assert config.barge_responding_threshold == 0.25
        # Non-str values are coerced, not dropped.
        config = TurnConfig.from_env(overrides={"stop_phrases": 123})
        assert config.stop_phrase_list == ["123"]


class TestFallbackAndManual:
    def test_gate_unavailable_commits_on_double_silence(self):
        """No semantic gate → commit after 2x min_silence_secs."""
        engine = TurnEngine(
            config=TurnConfig(semantic_enabled=False),
            vad=ScriptedVAD(),
            gate=None,
        )
        assert engine.semantic_active is False

        _, t = drive(engine, [0.9] * 10, 0.0)
        events, _ = drive(engine, [0.0] * frames_for(3.6), t)
        committed = [e for e in events if e.type == TURN_COMMITTED]
        assert len(committed) == 1
        assert committed[0].reason == REASON_TIMEOUT

    def test_force_commit_manual(self):
        engine = make_engine(FakeGate(0.1))
        _, t = drive(engine, [0.9] * 10, 0.0)
        events = engine.force_commit()
        assert len(events) == 1
        assert events[0].type == TURN_COMMITTED
        assert events[0].reason == REASON_MANUAL
        assert len(events[0].audio) > 0
        assert engine.state == TurnState.IDLE

    def test_force_commit_idle_is_noop(self):
        engine = make_engine(FakeGate(0.9))
        assert engine.force_commit() == []


class TestDiagnostics:
    def test_noise_floor_ema_tracks_nonspeech_frames(self):
        engine = make_engine(FakeGate(0.9))
        loud_noise = np.full(FRAME_SIZE, 0.05, dtype=np.float32)
        for i in range(20):
            engine.vad.push([0.1])  # scored as non-speech
            engine.feed(loud_noise, i * FRAME_SECS)
        diag = engine.diagnostics()
        assert diag["noise_floor_rms"] == pytest.approx(0.05, rel=0.05)
        assert diag["state"] == TurnState.IDLE.value

    def test_reset(self):
        engine = make_engine(FakeGate(0.9))
        drive(engine, [0.9] * 10, 0.0)
        engine.reset()
        assert engine.state == TurnState.IDLE
        assert engine.force_commit() == []


# --------------------------------------------------------------------------- #
# Real-model integration test
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not list(MODEL_DIR.glob("*.onnx")),
    reason="smart-turn ONNX model not downloaded "
    "(run: python scripts/download_models.py smart-turn)",
)
def test_real_smart_turn_gate():
    gate = SmartTurnGate()
    assert gate.available
    prob = gate.predict(np.zeros(16000, dtype=np.float32))
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

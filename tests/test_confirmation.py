import numpy as np
import pytest

from inference.confirmation import (
    ConfirmationConfig,
    ConfirmationEngine,
    compute_motion_score,
)


def test_candidate_to_confirmed_to_cooldown():
    engine = ConfirmationEngine(
        ConfirmationConfig(
            candidate_threshold=0.7,
            inactivity_seconds=2.0,
            motion_threshold=0.2,
            confirm_window_seconds=4.0,
            cooldown_seconds=3.0,
        )
    )
    assert engine.step(0.8, 0.5, 1.0) == "candidate"
    assert engine.step(0.4, 0.1, 1.0) == "candidate"
    assert engine.step(0.2, 0.1, 1.0) == "confirmed"
    assert engine.step(0.2, 0.1, 1.0) == "cooldown"


def test_candidate_times_out_without_sustained_low_motion():
    engine = ConfirmationEngine(
        ConfirmationConfig(
            candidate_threshold=0.7,
            inactivity_seconds=2.0,
            motion_threshold=0.2,
            confirm_window_seconds=2.0,
            cooldown_seconds=1.0,
        )
    )

    assert engine.step(0.8, 0.6, 1.0) == "candidate"
    assert engine.step(0.8, 0.6, 1.0) == "idle"


def test_compute_motion_score_uses_cross_window_delta():
    previous = np.full((52, 100), 10.0, dtype=np.float32)
    current = np.full((52, 100), 11.0, dtype=np.float32)

    assert compute_motion_score(current, previous) == pytest.approx(0.1)


def test_cooldown_exit_allows_immediate_candidate_on_same_step():
    """When cooldown expires and fall_score is high on the SAME step, the engine
    must return 'candidate' immediately — not make the caller wait an extra step.
    """
    engine = ConfirmationEngine(
        ConfirmationConfig(
            candidate_threshold=0.7,
            inactivity_seconds=2.0,
            motion_threshold=0.2,
            confirm_window_seconds=4.0,
            cooldown_seconds=1.0,
        )
    )
    # Drive into cooldown
    assert engine.step(0.8, 0.5, 1.0) == "candidate"    # idle -> candidate
    assert engine.step(0.4, 0.1, 1.0) == "candidate"    # accumulate inactivity
    assert engine.step(0.2, 0.1, 1.0) == "confirmed"    # 2 s inactivity -> confirmed
    assert engine.step(0.2, 0.1, 1.0) == "cooldown"     # confirmed -> cooldown (1 s left)

    # On the very step cooldown expires, fall_score is high again.
    # Expected: cooldown drains to 0, engine resets to idle, then immediately
    # detects the high fall_score and transitions to candidate.
    state = engine.step(0.9, 0.5, 1.0)
    assert state == "candidate", (
        f"Expected 'candidate' on cooldown-expiry step with high fall_score, got {state!r}"
    )


def test_cooldown_remains_while_timer_positive():
    """While _cooldown_left > 0, step must return 'cooldown'."""
    engine = ConfirmationEngine(
        ConfirmationConfig(
            candidate_threshold=0.5,
            inactivity_seconds=1.0,
            motion_threshold=0.2,
            confirm_window_seconds=5.0,
            cooldown_seconds=3.0,
        )
    )
    engine.step(0.9, 0.5, 1.0)  # candidate
    engine.step(0.9, 0.1, 1.0)  # confirmed (1 s inactivity)
    engine.step(0.9, 0.1, 1.0)  # cooldown (3 s left)
    assert engine.step(0.9, 0.1, 1.0) == "cooldown"   # 2 s left
    assert engine.step(0.9, 0.1, 1.0) == "cooldown"   # 1 s left
    # On the expiry step, even with low fall_score, must return 'idle'
    state = engine.step(0.1, 0.1, 1.0)
    assert state == "idle"


def test_cooldown_float_precision_expiry():
    """Cooldown must expire correctly even when dt does not divide cooldown_seconds evenly."""
    engine = ConfirmationEngine(
        ConfirmationConfig(
            candidate_threshold=0.5,
            inactivity_seconds=1.0,
            motion_threshold=0.2,
            confirm_window_seconds=5.0,
            cooldown_seconds=1.0,
        )
    )
    engine.step(0.9, 0.5, 1.0)  # candidate
    engine.step(0.9, 0.1, 1.0)  # confirmed
    engine.step(0.9, 0.1, 1.0)  # cooldown (1.0 s left)

    # Use dt=0.3; after 4 steps: 1.0 - 0.3 - 0.3 - 0.3 - 0.3 = -0.2 -> clamped to 0.
    # The <= 0.0 check (not == 0.0) must catch this.
    assert engine.step(0.1, 0.1, 0.3) == "cooldown"
    assert engine.step(0.1, 0.1, 0.3) == "cooldown"
    assert engine.step(0.1, 0.1, 0.3) == "cooldown"
    state = engine.step(0.1, 0.1, 0.3)   # _cooldown_left - 0.3 goes negative -> clamped 0
    assert state == "idle"

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

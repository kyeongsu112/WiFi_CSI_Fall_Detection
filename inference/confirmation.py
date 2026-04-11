from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfirmationConfig:
    candidate_threshold: float = 0.75
    inactivity_seconds: float = 6.0
    motion_threshold: float = 0.15
    confirm_window_seconds: float = 8.0
    cooldown_seconds: float = 10.0


def compute_motion_score(
    current_window: np.ndarray,
    previous_window: np.ndarray | None,
) -> float:
    """Return a normalized cross-window motion score.

    The baseline score is the mean absolute amplitude change between the
    current and previous windows, normalized by the previous window magnitude.
    Lower scores indicate less movement after the candidate fall.
    """
    if previous_window is None:
        return 1.0
    if current_window.shape != previous_window.shape:
        raise ValueError(
            "Motion score requires matching window shapes: "
            f"{current_window.shape} != {previous_window.shape}"
        )

    delta = float(np.mean(np.abs(current_window - previous_window)))
    baseline = float(np.mean(np.abs(previous_window)))
    if baseline <= 1e-6:
        return delta
    return delta / baseline


class ConfirmationEngine:
    def __init__(self, config: ConfirmationConfig):
        self.config = config
        self.state = "idle"
        self._inactive_accum = 0.0
        self._candidate_elapsed = 0.0
        self._cooldown_left = 0.0

    def step(self, fall_score: float, motion_score: float, dt: float) -> str:
        dt = max(0.0, dt)

        if self.state == "cooldown":
            self._cooldown_left = max(0.0, self._cooldown_left - dt)
            if self._cooldown_left <= 0.0:
                self._reset_to_idle()
                # fall through so a high fall_score on this same step can
                # immediately open a new candidate window
            else:
                return self.state

        if self.state == "idle" and fall_score >= self.config.candidate_threshold:
            self.state = "candidate"
            self._inactive_accum = 0.0
            self._candidate_elapsed = dt
            return self.state

        if self.state == "candidate":
            self._candidate_elapsed += dt

            if motion_score <= self.config.motion_threshold:
                self._inactive_accum += dt
                if self._inactive_accum >= self.config.inactivity_seconds:
                    self.state = "confirmed"
                    return self.state
            else:
                self._inactive_accum = 0.0
                if fall_score < self.config.candidate_threshold:
                    self._reset_to_idle()
                    return self.state

            if self._candidate_elapsed >= self.config.confirm_window_seconds:
                self._reset_to_idle()
            return self.state

        if self.state == "confirmed":
            self.state = "cooldown"
            self._cooldown_left = self.config.cooldown_seconds
            return self.state

        return self.state

    def _reset_to_idle(self) -> None:
        self.state = "idle"
        self._inactive_accum = 0.0
        self._candidate_elapsed = 0.0

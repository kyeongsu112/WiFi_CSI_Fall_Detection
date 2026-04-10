from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfirmationConfig:
    candidate_threshold: float = 0.75
    inactivity_seconds: float = 6.0
    motion_threshold: float = 0.15
    cooldown_seconds: float = 10.0


class ConfirmationEngine:
    def __init__(self, config: ConfirmationConfig):
        self.config = config
        self.state = "idle"
        self._inactive_accum = 0.0
        self._cooldown_left = 0.0

    def step(self, fall_score: float, motion_score: float, dt: float) -> str:
        if self.state == "cooldown":
            self._cooldown_left = max(0.0, self._cooldown_left - dt)
            if self._cooldown_left == 0.0:
                self.state = "idle"
            return self.state

        if self.state == "idle" and fall_score >= self.config.candidate_threshold:
            self.state = "candidate"
            self._inactive_accum = 0.0
            return self.state

        if self.state == "candidate":
            if motion_score <= self.config.motion_threshold:
                self._inactive_accum += dt
                if self._inactive_accum >= self.config.inactivity_seconds:
                    self.state = "confirmed"
                    return self.state
            else:
                self._inactive_accum = 0.0
                if fall_score < self.config.candidate_threshold:
                    self.state = "idle"
            return self.state

        if self.state == "confirmed":
            self.state = "cooldown"
            self._cooldown_left = self.config.cooldown_seconds
            return self.state

        return self.state

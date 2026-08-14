"""Command boundary for the read-only onboard milestone.

There is deliberately no Unitree LowCmd import in this module.  The only
available bridge validates commands; attempting to write is a hard failure.
"""

from typing import Sequence

import numpy as np


class ReadOnlyCommandBridge:
    """Validate a navigation command without exposing a robot write method."""

    def __init__(self, lower: Sequence[float], upper: Sequence[float], filter_alpha: float = 1.0):
        self.lower = np.asarray(lower, dtype=np.float32)
        self.upper = np.asarray(upper, dtype=np.float32)
        if self.lower.shape != (3,) or self.upper.shape != (3,):
            raise ValueError("command bounds must have shape (3,)")
        if np.any(self.lower > self.upper):
            raise ValueError("command lower bound exceeds upper bound")
        if not 0.0 < float(filter_alpha) <= 1.0:
            raise ValueError("filter_alpha must be in (0, 1]")
        self.filter_alpha = float(filter_alpha)
        self._filtered = np.zeros(3, dtype=np.float32)
        self._initialized = False
        self.validation_count = 0
        self.write_count = 0

    def validate(self, command: Sequence[float]) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32).reshape(-1)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command must be a finite 3-vector")
        self.validation_count += 1
        return np.clip(command, self.lower, self.upper)

    def filter(self, command: Sequence[float]) -> np.ndarray:
        command = self.validate(command)
        if not self._initialized:
            self._filtered = command.copy()
            self._initialized = True
        else:
            alpha = self.filter_alpha
            self._filtered = alpha * command + (1.0 - alpha) * self._filtered
        return self._filtered.copy()

    def reset(self) -> None:
        self._filtered.fill(0.0)
        self._initialized = False

    def send_low_level(self, _command) -> None:
        self.write_count += 1
        raise RuntimeError("low-level command output is disabled in sensor/shadow modes")

    def assert_no_writes(self) -> None:
        if self.write_count:
            raise AssertionError(f"unexpected low-level writes: {self.write_count}")

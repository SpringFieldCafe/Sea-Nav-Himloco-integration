"""Read-only runtime safety state machine."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

import numpy as np


class RuntimeState(str, Enum):
    INIT = "INIT"
    SENSOR_WAIT = "SENSOR_WAIT"
    SENSOR = "SENSOR"
    SHADOW = "SHADOW"
    STALE_SENSOR = "STALE_SENSOR"
    INVALID_DATA = "INVALID_DATA"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True)
class SafetyReport:
    state: RuntimeState
    finite: bool
    stale: bool
    deadline_miss: bool
    wireless_emergency: bool
    fault_reason: str


class SafetySupervisor:
    """Decides diagnostics state only; it never sends a robot command."""

    def __init__(self, max_sensor_age: float = 0.25):
        self.max_sensor_age = float(max_sensor_age)
        self.state = RuntimeState.INIT

    def evaluate(
        self,
        ages: Mapping[str, Optional[float]],
        values=(),
        mode: str = "shadow",
        deadline_miss: bool = False,
        wireless_emergency: bool = False,
    ) -> SafetyReport:
        finite = all(_is_finite(value) for value in values)
        stale = any(age is None or age > self.max_sensor_age for age in ages.values())
        if wireless_emergency:
            state, reason = RuntimeState.EMERGENCY_STOP, "wireless_emergency"
        elif not finite:
            state, reason = RuntimeState.INVALID_DATA, "nan_or_inf"
        elif stale:
            state, reason = RuntimeState.STALE_SENSOR, "stale_or_missing_sensor"
        elif deadline_miss:
            state, reason = RuntimeState.SHADOW, "deadline_miss"
        elif mode == "sensor":
            state, reason = RuntimeState.SENSOR, ""
        else:
            state, reason = RuntimeState.SHADOW, ""
        self.state = state
        return SafetyReport(state, finite, stale, bool(deadline_miss), bool(wireless_emergency), reason)


def _is_finite(value) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isfinite(np.asarray(value)).all())
    except (TypeError, ValueError):
        return True

import math
import time


class SafetyMonitor:
    """Diagnostics only; this class intentionally has no command publisher."""

    def __init__(self, max_sensor_age=0.25):
        self.max_sensor_age = max_sensor_age
        self.last_report = 0.0

    def check(self, ages, nav_action, him_action):
        values = list(ages.values()) + [nav_action, him_action]
        finite = all(_finite(value) for value in values)
        stale = any(age is None or age > self.max_sensor_age for age in ages.values())
        return {"finite": finite, "stale": stale, "ages": ages}

    def should_report(self, now=None):
        now = time.monotonic() if now is None else now
        if now - self.last_report >= 1.0:
            self.last_report = now
            return True
        return False


def _finite(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
    except ImportError:
        pass
    return True

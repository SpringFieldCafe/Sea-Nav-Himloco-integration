import time


class NumericStats:
    """Small bounded percentile accumulator for foreground diagnostics."""

    def __init__(self, limit=10000):
        self.limit = int(limit)
        self.values = []

    def add(self, value):
        self.values.append(float(value))
        if len(self.values) > self.limit:
            self.values.pop(0)

    def percentile(self, fraction):
        if not self.values:
            return 0.0
        values = sorted(self.values)
        index = (len(values) - 1) * float(fraction)
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        weight = index - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    def summary(self, prefix):
        return {
            f"{prefix}_mean_ms": sum(self.values) / len(self.values) if self.values else 0.0,
            f"{prefix}_p50_ms": self.percentile(0.50),
            f"{prefix}_p95_ms": self.percentile(0.95),
            f"{prefix}_p99_ms": self.percentile(0.99),
            f"{prefix}_max_ms": max(self.values) if self.values else 0.0,
        }


class RateStats:
    """Track inter-arrival periods and convert them to rate/percentile metrics."""

    def __init__(self, limit=10000):
        self.periods = []
        self.last_timestamp = None
        self.limit = int(limit)

    def observe(self, timestamp=None):
        timestamp = time.perf_counter() if timestamp is None else float(timestamp)
        if self.last_timestamp is not None:
            period = timestamp - self.last_timestamp
            if period > 0.0:
                self.periods.append(period)
                if len(self.periods) > self.limit:
                    self.periods.pop(0)
        self.last_timestamp = timestamp

    def current_rate_hz(self):
        if not self.periods:
            return 0.0
        return 1.0 / (sum(self.periods) / len(self.periods))

    def summary(self):
        if not self.periods:
            return {
                "samples": 0, "mean_rate_hz": 0.0, "p50_period_ms": 0.0,
                "p95_period_ms": 0.0, "p99_period_ms": 0.0, "max_period_ms": 0.0,
            }
        stats = NumericStats(limit=self.limit)
        for period in self.periods:
            stats.add(period * 1000.0)
        mean_period = sum(self.periods) / len(self.periods)
        return {
            "samples": len(self.periods),
            "mean_rate_hz": 1.0 / mean_period,
            "p50_period_ms": stats.percentile(0.50),
            "p95_period_ms": stats.percentile(0.95),
            "p99_period_ms": stats.percentile(0.99),
            "max_period_ms": max(self.periods) * 1000.0,
        }


class FixedRate:
    def __init__(self, hz):
        self.period = 1.0 / hz
        self.next_deadline = time.monotonic()

    def sleep(self):
        self.next_deadline += self.period
        delay = self.next_deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_deadline = time.monotonic()

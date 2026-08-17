"""Read-only scheduler probe for separating Python, SDK, and DDS effects."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import statistics
import threading
import time
from collections import deque


PERIOD = 0.020


def percentile(values, p):
    if not values:
        return 0.0
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1] if len(values) > 1 else values[0]


def system_snapshot():
    governors = {}
    frequencies = {}
    for path in pathlib.Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"):
        try:
            governors[path.parent.name] = path.read_text().strip()
        except OSError:
            pass
    for path in pathlib.Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_cur_freq"):
        try:
            frequencies[path.parent.name] = int(path.read_text().strip())
        except (OSError, ValueError):
            pass
    with open("/proc/loadavg", encoding="utf-8") as handle:
        load = handle.read().split()[:3]
    try:
        scheduler = os.sched_getscheduler(0)
    except OSError:
        scheduler = None
    scheduler_names = {0: "SCHED_OTHER", 1: "SCHED_FIFO", 2: "SCHED_RR", 3: "SCHED_BATCH", 5: "SCHED_IDLE", 6: "SCHED_DEADLINE"}
    return {
        "cpu_governor": governors,
        "cpu_frequency_khz": frequencies,
        "loadavg": load,
        "pid": os.getpid(),
        "scheduler_class": scheduler,
        "scheduler_class_name": scheduler_names.get(scheduler, "unknown"),
    }


class Heartbeat:
    def __init__(self):
        self.stop = threading.Event()
        self.intervals = deque(maxlen=100000)
        self.thread = threading.Thread(target=self._run, name="scheduler-probe-heartbeat", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        previous = time.monotonic()
        while not self.stop.wait(0.005):
            current = time.monotonic()
            self.intervals.append(current - previous)
            previous = current

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.0)


def _run_scheduler(duration, on_tick=None):
    """Run the common 50 Hz loop while an optional read-only callback runs."""
    heartbeat = Heartbeat()
    heartbeat.start()
    starts = []
    periods = []
    overshoots = []
    deadline_misses = 0
    started_at = time.monotonic()
    next_tick = started_at
    previous_start = None
    end = started_at + duration
    while time.monotonic() < end:
        loop_start = time.monotonic()
        starts.append(loop_start)
        if previous_start is not None:
            periods.append(loop_start - previous_start)
        previous_start = loop_start
        if on_tick is not None:
            on_tick()

        next_tick += PERIOD
        requested = max(0.0, next_tick - time.monotonic())
        sleep_start = time.monotonic()
        if requested > 0.0:
            time.sleep(requested)
        sleep_actual = time.monotonic() - sleep_start
        overshoots.append(max(0.0, sleep_actual - requested))
        if requested == 0.0:
            deadline_misses += 1
            next_tick = time.monotonic()
    elapsed = time.monotonic() - started_at
    heartbeat.close()

    def ms(values):
        return [value * 1000.0 for value in values]

    period_ms = ms(periods)
    overshoot_ms = ms(overshoots)
    heartbeat_ms = ms(list(heartbeat.intervals))
    return {
        "duration_s": duration,
        "elapsed_s": elapsed,
        "samples": len(starts),
        "period_ms": stats(period_ms),
        "sleep_overshoot_ms": stats(overshoot_ms),
        "heartbeat_gap_ms": stats(heartbeat_ms),
        "period_gt20_count": sum(value > 20.0 for value in period_ms),
        "sleep_overshoot_gt20_count": sum(value > 20.0 for value in overshoot_ms),
        "heartbeat_gt20_count": sum(value > 20.0 for value in heartbeat_ms),
        "heartbeat_gt50_count": sum(value > 50.0 for value in heartbeat_ms),
        "heartbeat_gt100_count": sum(value > 100.0 for value in heartbeat_ms),
        "deadline_misses": deadline_misses,
    }


def run_probe(duration, mode, net=None):
    if mode == "sdk_import_only":
        importlib.import_module("unitree_sdk2py.core.channel")
    elif mode == "sdk_factory_only":
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(0, None)
    result = _run_scheduler(duration)
    result.update({"mode": mode, "system": system_snapshot()})
    return result


def run_lowstate_probe(duration, net):
    """Measure the scheduler with only the read-only rt/lowstate subscriber."""
    channel = importlib.import_module("unitree_sdk2py.core.channel")
    messages = importlib.import_module("unitree_sdk2py.idl.unitree_go.msg.dds_")

    # Deliberately import only the factory/subscriber and state type here.
    # The D path contains no write-side type, method, or control client.
    channel.ChannelFactoryInitialize(0, net)
    lowstate_type = messages.LowState_
    measurements = {
        "rx_count": 0,
        "last_rx": None,
        "gaps": [],
        "callback_durations": [],
    }

    def callback(_message):
        started = time.monotonic()
        previous = measurements["last_rx"]
        received_at = time.monotonic()
        measurements["rx_count"] += 1
        measurements["last_rx"] = received_at
        if previous is not None:
            measurements["gaps"].append(received_at - previous)
        measurements["callback_durations"].append(time.monotonic() - started)

    subscriber = channel.ChannelSubscriber("rt/lowstate", lowstate_type)
    subscriber.Init(callback, 10)
    try:
        result = _run_scheduler(duration)
    finally:
        subscriber.Close()

    elapsed = result["elapsed_s"]
    gaps_ms = [value * 1000.0 for value in measurements["gaps"]]
    callback_ms = [value * 1000.0 for value in measurements["callback_durations"]]
    result.update(
        {
            "mode": "lowstate_read_only",
            "topic": "rt/lowstate",
            "queue_len": 10,
            "lowstate": {
                "rx_count": measurements["rx_count"],
                "rx_rate_hz": measurements["rx_count"] / elapsed if elapsed else 0.0,
                "rx_gap_p50_ms": percentile(gaps_ms, 50) if gaps_ms else 0.0,
                "rx_gap_p95_ms": percentile(gaps_ms, 95) if gaps_ms else 0.0,
                "rx_gap_p99_ms": percentile(gaps_ms, 99) if gaps_ms else 0.0,
                "rx_gap_max_ms": max(gaps_ms) if gaps_ms else 0.0,
                "callback_p95_ms": percentile(callback_ms, 95) if callback_ms else 0.0,
                "callback_max_ms": max(callback_ms) if callback_ms else 0.0,
            },
            "LOWCMD_PUBLISHER_CREATED": "NO",
            "LOWCMD_WRITE_COUNT": 0,
            "system": system_snapshot(),
        }
    )
    return result


def stats(values):
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pure_python", "sdk_import_only", "sdk_factory_only", "lowstate_read_only"), required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--net", default=None, help="network interface used only by lowstate_read_only")
    parser.add_argument("--allow-live-lowstate", action="store_true")
    args = parser.parse_args()
    if args.mode == "lowstate_read_only" and not args.allow_live_lowstate:
        raise SystemExit("lowstate_read_only is live-network guarded; pass --allow-live-lowstate explicitly")
    if args.mode == "lowstate_read_only":
        if not args.net:
            raise SystemExit("lowstate_read_only requires --net")
        result = run_lowstate_probe(args.duration, args.net)
    else:
        result = run_probe(args.duration, args.mode, args.net)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

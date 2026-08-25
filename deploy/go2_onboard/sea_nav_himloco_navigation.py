"""SEA-Nav plus HIMLoco 1460 Go2 navigation controller.

The ROS sensor bridge remains a separate process.  This entry point starts a
Torch-only navigation worker process which consumes the bridge socket at about
10 Hz and publishes only the latest bounded command to the 50 Hz HIMLoco
controller in this process.  A stale or failed navigation worker becomes a
zero command; it can never keep the last non-zero command alive indefinitely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import socket
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .command_bridge import ReadOnlyCommandBridge
from .himloco_fixed_control import (
    ARM_WAIT_TIMEOUT,
    CONTROL_DT,
    DEFAULT_ANGLES,
    FixedCommand,
    FixedHIMLocoController,
    POSE_KD,
    POSE_KP,
    RuntimeState,
    SafetyError,
    configure_torch_runtime,
    policy_profile_for_path,
)
from .ipc_schema import decode_packet
from .model_loader import infer, load_navigation_policy
from .navigation_observation import NavigationObservation, clear_lidar_observation


DEFAULT_NAVIGATION_POLICY = "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt"
DEFAULT_NAVIGATION_METADATA = "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json"
DEFAULT_SENSOR_SOCKET = "/tmp/sea_nav_shadow.sock"
DEFAULT_NAVIGATION_LOG = "logs/go2_navigation/navigation.jsonl"
SEA_SHA256 = "d1242c74ff56189f20d7a12948d86078287651cd1f70215d9c308d04f4b561de"
HIM_1460_SHA256 = "cab2489dda7732a7d6f51595aa6362384445c738537d0c1c91569054c7b9f5d1"
NAV_LOWER = np.asarray([0.0, -np.inf, -np.inf], dtype=np.float32)
NAV_UPPER = np.asarray([np.inf, 0.0, np.inf], dtype=np.float32)
NAV_VX_MAX = float("inf")
NAV_VY_MAX = float("inf")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_navigation_model(path: str, metadata_path: str, device: str = "cpu"):
    """Validate SEA file content and 550->3 contract before connecting Go2."""
    resolved = str(Path(path).expanduser().resolve())
    actual = sha256_file(resolved)
    if actual != SEA_SHA256:
        raise SafetyError(f"SEA-Nav SHA256 mismatch: expected {SEA_SHA256}, got {actual}")
    loaded = load_navigation_policy(resolved, metadata_path, torch.device(device))
    if loaded.sha256 != SEA_SHA256:
        raise SafetyError("SEA-Nav loaded hash changed during validation")
    return loaded


def validate_navigation_himloco_policy(path: str, allow_override: bool = False) -> str:
    """Require 1460 by default; allow only explicit policy identity override."""
    resolved = str(Path(path).expanduser().resolve())
    profile = policy_profile_for_path(resolved, allow_override=allow_override)
    if profile != "himloco_1460" and not allow_override:
        raise SafetyError("SEA-Nav navigation entry requires the approved HIMLoco 1460 model")
    return sha256_file(resolved)


class NavigationLimiter:
    """Limit navigation velocity commands for an explicit test profile."""

    def __init__(self, filter_alpha: float = 0.15, vx_max: float = NAV_VX_MAX,
                 vy_max: float = 0.0):
        vx_max = float(vx_max)
        vy_max = float(vy_max)
        if not np.isfinite(vx_max) and not np.isinf(vx_max):
            raise ValueError("navigation vx max must be non-negative")
        if not np.isfinite(vy_max) and not np.isinf(vy_max):
            raise ValueError("navigation vy max must be non-negative")
        if vx_max < 0.0 or vy_max < 0.0:
            raise ValueError("navigation velocity limits must be non-negative")
        upper = NAV_UPPER.copy()
        upper[0] = vx_max
        lower = NAV_LOWER.copy()
        lower[1] = -vy_max
        upper[1] = vy_max
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.bridge = ReadOnlyCommandBridge(lower, upper, filter_alpha)
        self.lower = lower
        self.upper = upper

    def reset(self):
        self.bridge.reset()

    def apply(self, raw: Sequence[float]):
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise FloatingPointError("SEA-Nav command must be a finite 3-vector")
        limited = np.clip(raw, self.lower, self.upper)
        safe = self.bridge.filter(limited)
        return raw.copy(), limited, safe


def goal_is_reached(goal_body: Sequence[float], tolerance: float) -> bool:
    goal = np.asarray(goal_body, dtype=np.float32).reshape(-1)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise FloatingPointError("goal_body must be a finite 2-vector")
    return float(np.linalg.norm(goal)) <= float(tolerance)


def _packet_is_fresh(packet, freshness):
    validity = packet.get("validity", {})
    if not all(validity.get(name, False) for name in ("lowstate", "lidar", "odom", "goal")):
        return False, "invalid_or_missing_sensor"
    for name in ("lowstate", "odom", "lidar"):
        age = packet.get("sensor_age", {}).get(name)
        if age is None or float(age) > freshness[name]:
            return False, "stale_sensor"
    return True, ""


def _navigation_arg(args, name):
    return args.get(name) if isinstance(args, dict) else getattr(args, name)


def _navigation_result(
    packet, sea, nav_obs, limiter, previous_command, args, reached,
    goal_reached_streak=0,
):
    source_metadata = {
        "sensor_sequence": int(packet.get("sequence", -1)),
        "sensor_timestamp_monotonic": float(packet.get("timestamp_monotonic", 0.0)),
    }
    assume_clear_lidar = (args.get("assume_clear_lidar", False)
                          if isinstance(args, dict) else args.assume_clear_lidar)
    fresh, reason = _packet_is_fresh(packet, {
        "lowstate": _navigation_arg(args, "lowstate_max_age"),
        "odom": _navigation_arg(args, "odom_max_age"),
        "lidar": _navigation_arg(args, "lidar_max_age"),
    })
    goal_body = np.asarray(packet.get("goal_body", [0.0, 0.0]), dtype=np.float32)
    distance = float(np.linalg.norm(goal_body))
    goal_candidate = fresh and goal_is_reached(
        goal_body, _navigation_arg(args, "goal_tolerance")
    )
    confirmations = int(_navigation_arg(args, "goal_reached_confirmations"))
    if confirmations < 1:
        raise ValueError("goal_reached_confirmations must be positive")
    goal_reached_streak = goal_reached_streak + 1 if goal_candidate else 0
    if reached or goal_reached_streak >= confirmations:
        return {
            **source_metadata,
            "command": [0.0, 0.0, 0.0],
            "raw_command": [0.0, 0.0, 0.0],
            "limited_command": [0.0, 0.0, 0.0],
            "goal_body": goal_body.tolist(),
            "goal_distance": distance,
            "runtime_state": "GOAL_REACHED",
            "fault_reason": "",
            "sea_observation_shape": [1, 550],
            "sea_inference_latency_ms": 0.0,
            "sequence": int(packet["sequence"]),
            "timestamp_monotonic": time.monotonic(),
            "lowcmd_sent": False,
        }, True, goal_reached_streak
    if not fresh:
        nav_obs.reset()
        limiter.reset()
        return {
            **source_metadata,
            "command": [0.0, 0.0, 0.0],
            "raw_command": [0.0, 0.0, 0.0],
            "limited_command": [0.0, 0.0, 0.0],
            "goal_body": goal_body.tolist(),
            "goal_distance": distance,
            "runtime_state": "STALE_NAVIGATION",
            "fault_reason": reason,
            "sea_observation_shape": None,
            "sea_inference_latency_ms": 0.0,
            "sequence": int(packet["sequence"]),
            "timestamp_monotonic": time.monotonic(),
            "lowcmd_sent": False,
        }, False, 0

    device = torch.device("cpu")
    packet_tensor = {
        key: torch.as_tensor(packet[key], dtype=torch.float32, device=device).reshape(1, -1)
        for key in (
            "projected_gravity",
            "base_linear_velocity_body",
            "base_angular_velocity_body",
            "lidar_rays",
            "goal_body",
        )
    }
    command_tensor = torch.as_tensor(previous_command, dtype=torch.float32, device=device).reshape(1, 3)
    if assume_clear_lidar:
        packet_tensor["lidar_rays"] = torch.full_like(packet_tensor["lidar_rays"], 5.0)
    observation = nav_obs.build(
        packet_tensor["projected_gravity"],
        command_tensor,
        packet_tensor["base_linear_velocity_body"],
        packet_tensor["base_angular_velocity_body"],
        packet_tensor["lidar_rays"],
        packet_tensor["goal_body"],
    )
    if assume_clear_lidar:
        observation = clear_lidar_observation(observation)
    inference_start = time.perf_counter()
    raw = infer(sea, observation, 3)[0].detach().cpu().numpy()
    inference_ms = (time.perf_counter() - inference_start) * 1000.0
    raw_command, limited, safe = limiter.apply(raw)
    return {
        **source_metadata,
        "command": safe.tolist(),
        "raw_command": raw_command.tolist(),
        "limited_command": limited.tolist(),
        "goal_body": goal_body.tolist(),
        "goal_distance": distance,
        "runtime_state": "NAVIGATION",
        "fault_reason": "",
        "sea_observation_shape": list(observation.shape),
        "sea_observation_min": float(observation.min().item()),
        "sea_observation_max": float(observation.max().item()),
        "sea_inference_latency_ms": inference_ms,
        "sequence": int(packet["sequence"]),
        "timestamp_monotonic": time.monotonic(),
        "lowcmd_sent": False,
    }, False, goal_reached_streak


def _navigation_worker_main(config, sender, stop_event):
    """Run SEA inference away from the 50 Hz LowCmd process."""
    sock = None
    log = None
    try:
        configure_torch_runtime(1, 1)
        sea = load_navigation_policy(
            config["navigation_policy"],
            config["navigation_metadata"],
            torch.device("cpu"),
        )
        if sea.sha256 != SEA_SHA256:
            raise SafetyError("SEA-Nav worker hash mismatch")
        nav_obs = NavigationObservation(torch.device("cpu"))
        limiter = NavigationLimiter(
            config["command_filter_alpha"], config["navigation_vx_max"],
            config["navigation_vy_max"],
        )
        previous_command = np.zeros(3, dtype=np.float32)
        reached = False
        goal_reached_streak = 0
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(float(config["connect_timeout"]))
        sock.connect(config["sensor_socket"])
        sock.settimeout(0.2)
        buffer = b""
        latest_packet = None
        next_inference = 0.0
        if config["navigation_log"]:
            output = Path(config["navigation_log"])
            output.parent.mkdir(parents=True, exist_ok=True)
            log = output.open("a", encoding="utf-8")
        last_summary = time.monotonic()
        count = 0
        while not stop_event.is_set():
            try:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("sensor bridge disconnected")
                buffer += chunk
            except socket.timeout:
                chunk = b""
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line:
                    latest_packet = decode_packet(line)
            now = time.monotonic()
            if latest_packet is None or now < next_inference:
                continue
            result, reached, goal_reached_streak = _navigation_result(
                latest_packet, sea, nav_obs, limiter, previous_command, config, reached,
                goal_reached_streak,
            )
            previous_command = np.asarray(result["command"], dtype=np.float32)
            result["worker_send_timestamp_monotonic"] = time.monotonic()
            result["timestamp_monotonic"] = time.monotonic()
            sender.send(result)
            count += 1
            if log:
                log.write(json.dumps(result, separators=(",", ":")) + "\n")
            if now - last_summary >= float(config["summary_interval"]):
                print(
                    "[navigation] "
                    + json.dumps(
                        {
                            "state": result["runtime_state"],
                            "goal_body": result["goal_body"],
                            "goal_distance": result["goal_distance"],
                            "raw": result["raw_command"],
                            "safe": result["command"],
                            "sea_p95_not_available": True,
                            "samples": count,
                            "lowcmd_sent": False,
                        },
                        separators=(",", ":"),
                    )
                )
                last_summary = now
            next_inference = now + 1.0 / float(config["navigation_hz"])
        if log:
            log.close()
            log = None
        sock.close()
        sock = None
    except Exception as exc:
        try:
            sender.send({
                "runtime_state": "NAVIGATION_FAILED",
                "fault_reason": str(exc),
                "command": [0.0, 0.0, 0.0],
                "raw_command": [0.0, 0.0, 0.0],
                "limited_command": [0.0, 0.0, 0.0],
                "goal_body": [0.0, 0.0],
                "goal_distance": 0.0,
                "reader_exit_reason": "worker_exception",
                "socket_state": "worker_exception",
                "worker_error": repr(exc),
                "timestamp_monotonic": time.monotonic(),
                "lowcmd_sent": False,
            })
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if log:
            log.close()
        if sock is not None:
            sock.close()


class NavigationMailbox:
    """Thread-safe latest-command mailbox with a fail-closed stale result."""

    def __init__(self, max_age: float):
        self.max_age = float(max_age)
        self.condition = threading.Condition(threading.RLock())
        self.lock = self.condition
        self.command = np.zeros(3, dtype=np.float32)
        self.received_at = 0.0
        self.sensor_sequence = -1
        self.sensor_timestamp_monotonic = 0.0
        self.worker_send_timestamp_monotonic = 0.0
        self.mailbox_update_monotonic = 0.0
        self.last_handoff = {}
        self.last_result = {"runtime_state": "WAITING", "fault_reason": "navigation_not_ready"}

    def update(self, result, received_at=None):
        command = np.asarray(result.get("command", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        if command.shape != (3,) or not np.isfinite(command).all():
            command = np.zeros(3, dtype=np.float32)
            result = dict(result, runtime_state="NAVIGATION_FAILED", fault_reason="invalid_command")
        with self.lock:
            self.command = command
            self.received_at = time.monotonic()
            self.sensor_sequence = int(result.get("sensor_sequence", -1))
            self.sensor_timestamp_monotonic = float(
                result.get("sensor_timestamp_monotonic", 0.0) or 0.0
            )
            self.worker_send_timestamp_monotonic = float(
                result.get("worker_send_timestamp_monotonic", 0.0) or 0.0
            )
            self.mailbox_update_monotonic = (
                time.monotonic() if received_at is None else float(received_at)
            )
            self.last_result = dict(result)
            self.condition.notify_all()

    def current(self):
        with self.lock:
            delivery_age = time.monotonic() - self.received_at if self.received_at else float("inf")
            sensor_age = (
                time.monotonic() - self.sensor_timestamp_monotonic
                if self.sensor_timestamp_monotonic
                else float("inf")
            )
            age = max(delivery_age, sensor_age)
            if delivery_age > self.max_age or sensor_age > self.max_age:
                return FixedCommand(0.0, 0.0, 0.0), "navigation_command_stale", age
            return FixedCommand(*self.command.tolist()), "", age

    def wait_for_fresh_command(self, timeout: float, on_wait=None):
        """Return only a result sourced after this call and within max_age."""
        deadline = time.monotonic() + float(timeout)
        with self.lock:
            baseline_sequence = self.sensor_sequence
            self.command.fill(0.0)
        while True:
            now = time.monotonic()
            with self.lock:
                source_sequence = self.sensor_sequence
                source_timestamp = self.sensor_timestamp_monotonic
                received_at = self.received_at
                command = self.command.copy()
                state = self.last_result.get("runtime_state")
            source_age = now - source_timestamp if source_timestamp else float("inf")
            received_age = now - received_at if received_at else float("inf")
            if (
                source_sequence > baseline_sequence
                and source_age <= self.max_age
                and received_age <= self.max_age
                and state in ("NAVIGATION", "GOAL_REACHED")
            ):
                self.last_handoff = {
                    "boundary_sequence": baseline_sequence,
                    "sensor_sequence": source_sequence,
                    "source_age_ms": source_age * 1000.0,
                    "delivery_age_ms": received_age * 1000.0,
                    "worker_send_timestamp_monotonic": self.worker_send_timestamp_monotonic,
                    "mailbox_update_monotonic": self.mailbox_update_monotonic,
                }
                return FixedCommand(*command.tolist())
            if now >= deadline:
                raise SafetyError(
                    "no fresh navigation command after ACTIVE handoff "
                    f"boundary_sequence={baseline_sequence} "
                    f"current_sequence={source_sequence} "
                    f"source_age_s={source_age:.3f} "
                    f"delivery_age_s={received_age:.3f}"
                )
            if on_wait is not None:
                on_wait()
            with self.condition:
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    self.condition.wait(timeout=min(CONTROL_DT, remaining))


class NavigationProcess:
    def __init__(self, config, mailbox):
        self.config = dict(config)
        self.mailbox = mailbox
        self.context = mp.get_context("spawn")
        self.stop_event = self.context.Event()
        self.receiver, self.sender = self.context.Pipe(duplex=False)
        self.process = None
        self.receiver_thread = None
        self.receiver_error = None
        self.reader_exit_reason = None

    def start(self):
        self.process = self.context.Process(
            target=_navigation_worker_main,
            args=(self.config, self.sender, self.stop_event),
            name="sea-nav-worker",
        )
        self.process.start()
        self.receiver_thread = threading.Thread(target=self._receive, name="sea-nav-mailbox", daemon=True)
        self.receiver_thread.start()

    def _receive(self):
        while not self.stop_event.is_set():
            try:
                if self.receiver.poll(0.1):
                    self.mailbox.update(self.receiver.recv(), received_at=time.monotonic())
                elif self.process is not None and not self.process.is_alive():
                    if self.stop_event.is_set():
                        self.reader_exit_reason = "normal_shutdown"
                    else:
                        self.reader_exit_reason = "worker_exit"
                        self.receiver_error = "navigation worker exited unexpectedly"
                    return
            except (EOFError, OSError) as exc:
                self.receiver_error = str(exc)
                self.reader_exit_reason = "reader_exception"
                return

    def status(self):
        return {
            "worker_alive": bool(self.process is not None and self.process.is_alive()),
            "reader_thread_alive": bool(
                self.receiver_thread is not None and self.receiver_thread.is_alive()
            ),
            "reader_error": self.receiver_error,
            "reader_exit_reason": self.reader_exit_reason,
            "last_sequence": self.mailbox.sensor_sequence,
            "last_packet_age_s": (
                time.monotonic() - self.mailbox.sensor_timestamp_monotonic
                if self.mailbox.sensor_timestamp_monotonic
                else float("inf")
            ),
            "socket_state": self.mailbox.last_result.get("socket_state", "unknown"),
        }

    def stop(self):
        self.stop_event.set()
        if self.process is not None:
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)
        try:
            self.receiver.close()
            self.sender.close()
        except OSError:
            pass


class SeaNavHimLocoController(FixedHIMLocoController):
    """Reuse the guarded LowCmd controller with a latest SEA command."""

    def __init__(self, args, navigation):
        super().__init__(args)
        self.navigation = navigation
        self.last_navigation_command = np.zeros(3, dtype=np.float32)
        self.last_navigation_command_age_s = float("inf")
        self.last_navigation_action_norm = 0.0
        self._last_navigation_summary = 0.0
        self._last_navigation_fault = None

    def _current_navigation_command(self):
        command, reason, age = self.navigation.mailbox.current()
        if reason and reason != self._last_navigation_fault:
            print(f"[safety] navigation command forced to zero: reason={reason} age_s={age:.3f}")
            self._last_navigation_fault = reason
        elif not reason:
            self._last_navigation_fault = None
        return command

    def _hold_default_pose_while_waiting_for_navigation(self):
        snapshot = self._snapshot()
        self._check_runtime_safety(snapshot)
        self._send_timed_pose_command(DEFAULT_ANGLES, POSE_KP, POSE_KD)

    def _initialize_policy_history(self, snapshot, _command):
        command = self.navigation.mailbox.wait_for_fresh_command(
            ARM_WAIT_TIMEOUT,
            on_wait=self._hold_default_pose_while_waiting_for_navigation,
        )
        with self.navigation.mailbox.lock:
            handoff = dict(self.navigation.mailbox.last_handoff)
        print(
            "[navigation] ACTIVE_HANDOFF_FRESH "
            + json.dumps(handoff, separators=(",", ":"))
        )
        return super()._initialize_policy_history(snapshot, command)

    def _handle_policy_arm_failure(self, error):
        if "navigation" not in str(error).lower():
            return False
        self.state = RuntimeState.DEFAULT_POSE_HOLD
        status = self.navigation.status()
        print(f"[safety] NAVIGATION_ARM_FAILED: {error}")
        print(
            "[navigation] "
            "worker_alive=%s reader_thread_alive=%s reader_error=%s "
            "reader_exit_reason=%s last_sequence=%s last_packet_age_s=%.3f socket_state=%s"
            % (
                status["worker_alive"],
                status["reader_thread_alive"],
                status["reader_error"],
                status["reader_exit_reason"],
                status["last_sequence"],
                status["last_packet_age_s"],
                status["socket_state"],
            )
        )
        print("[safety] DEFAULT_POSE_HOLD: release A; press A to retry; Select=STOP; B=ESTOP; Ctrl+C=STOP")
        return True

    def _run_policy_once(self, _command, snapshot, observation=None):
        navigation_command = self._current_navigation_command()
        self.last_navigation_command = navigation_command.as_array().copy()
        with self.navigation.mailbox.lock:
            result = dict(self.navigation.mailbox.last_result)
        sent_at = result.get("timestamp_monotonic")
        self.last_navigation_command_age_s = (
            max(0.0, time.monotonic() - float(sent_at))
            if sent_at is not None else float("inf")
        )
        action = super()._run_policy_once(
            navigation_command,
            snapshot,
            observation=observation,
        )
        self.last_navigation_action_norm = float(np.linalg.norm(action))
        return action

    def _print_diagnostics(self, now):
        super()._print_diagnostics(now)
        if now - self._last_navigation_summary < 1.0:
            return
        self._last_navigation_summary = now
        with self.navigation.mailbox.lock:
            result = dict(self.navigation.mailbox.last_result)
        print(
            "[navigation] latest="
            + json.dumps(
                {
                    "state": result.get("runtime_state", "UNKNOWN"),
                    "goal_body": result.get("goal_body"),
                    "goal_distance": result.get("goal_distance"),
                    "raw": result.get("raw_command"),
                    "safe": result.get("command"),
                    "lowcmd_sent": False,
                },
                separators=(",", ":"),
            )
        )
        print(
            "[navigation-control] "
            f"command={self.last_navigation_command.tolist()} "
            f"command_age_s={self.last_navigation_command_age_s:.4f} "
            f"action_norm={self.last_navigation_action_norm:.4f} "
            f"lowcmd_publish_count={self.lowcmd_publish_count} "
            f"active_lowcmd_publish_count={self.active_lowcmd_publish_count}"
        )

    def stop(self):
        self.navigation.stop()
        super().stop()


def build_parser():
    from .himloco_fixed_control import build_parser as build_fixed_parser

    parser = build_fixed_parser()
    parser.description = "SEA-Nav 550->3 plus guarded HIMLoco 1460 Go2 navigation controller"
    parser.add_argument("--sensor-socket", default=DEFAULT_SENSOR_SOCKET)
    parser.add_argument("--navigation-policy", default=DEFAULT_NAVIGATION_POLICY)
    parser.add_argument("--navigation-metadata", default=DEFAULT_NAVIGATION_METADATA)
    parser.add_argument("--navigation-hz", type=float, default=10.0)
    parser.add_argument("--navigation-command-max-age", type=float, default=0.25)
    parser.add_argument("--navigation-filter-alpha", type=float, default=0.15)
    parser.add_argument(
        "--navigation-vx-max",
        type=float,
        default=NAV_VX_MAX,
        help="navigation forward-vx upper bound in m/s; inf disables the speed limit",
    )
    parser.add_argument(
        "--navigation-vy-max",
        type=float,
        default=0.0,
        help="optional lateral-vy bound in m/s; inf disables the lateral speed limit",
    )
    parser.add_argument("--navigation-connect-timeout", type=float, default=10.0)
    parser.add_argument("--navigation-summary-interval", type=float, default=1.0)
    parser.add_argument(
        "--odom-max-age", type=float, default=0.10,
        help="maximum odometry age before navigation forces a zero command",
    )
    parser.add_argument("--assume-clear-lidar", action="store_true",
                        help="NO OBSTACLE AVOIDANCE: use 5m LiDAR rays")
    parser.add_argument(
        "--allow-policy-override",
        action="store_true",
        help="explicit test mode: allow non-1460 HIMLoco policies after the 270->12 contract gate",
    )
    parser.add_argument("--navigation-log", default=DEFAULT_NAVIGATION_LOG)
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--front-goal-distance", type=float)
    parser.add_argument("--goal-tolerance", type=float, default=0.15)
    parser.add_argument("--goal-reached-confirmations", type=int, default=5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.legacy_repro or args.comm_only:
        raise SystemExit("SEA-Nav navigation only supports the HIMLoco 1460 active profile")
    if not 0.0 < args.navigation_hz <= 20.0:
        raise SystemExit("--navigation-hz must be in (0,20]")
    if (args.goal_x is None) != (args.goal_y is None):
        raise SystemExit("provide both --goal-x and --goal-y, or neither")
    if args.front_goal_distance is not None and (args.goal_x is not None or args.goal_y is not None):
        raise SystemExit("front-goal-distance cannot be combined with goal-x/goal-y")
    if (args.navigation_command_max_age <= 0.0 or args.goal_tolerance <= 0.0
            or args.odom_max_age <= 0.0 or args.max_sensor_age <= 0.0):
        raise SystemExit(
            "navigation freshness, LowState age, odom age, and goal tolerance must be positive"
        )
    if args.goal_reached_confirmations < 1:
        raise SystemExit("--goal-reached-confirmations must be positive")
    if (not np.isfinite(args.navigation_vx_max) and not np.isinf(args.navigation_vx_max)) or args.navigation_vx_max < 0.0:
        raise SystemExit("--navigation-vx-max must be non-negative or inf")
    if (not np.isfinite(args.navigation_vy_max) and not np.isinf(args.navigation_vy_max)) or args.navigation_vy_max < 0.0:
        raise SystemExit("--navigation-vy-max must be non-negative or inf")
    # The fixed controller parser is reused for its transport/safety options.
    # Navigation owns the command source; non-zero fixed commands are rejected
    # instead of silently becoming a second command path.
    if any(abs(float(value)) > 1e-8 for value in (args.vx, args.vy, args.wz)):
        raise SystemExit("navigation entry requires fixed command placeholders --vx 0 --vy 0 --wz 0")
    configure_torch_runtime(args.torch_threads, args.torch_interop_threads)
    sea = validate_navigation_model(args.navigation_policy, args.navigation_metadata)
    print(f"[model] SEA path={sea.path} sha256={sea.sha256} input=550 output=3")
    himloco_sha = validate_navigation_himloco_policy(
        args.policy, allow_override=args.allow_policy_override
    )
    if args.allow_policy_override:
        print(f"[safety] HIMLOCO_POLICY_OVERRIDE=ENABLED sha256={himloco_sha}")
    if args.front_goal_distance is not None:
        print(f"[navigation] FRONT_GOAL_DISTANCE={args.front_goal_distance:.6f}m frame=odom")
    else:
        print(f"[navigation] GOAL_WORLD=[{args.goal_x:.6f},{args.goal_y:.6f}] frame=odom")
    print(f"[navigation] GOAL_TOLERANCE={args.goal_tolerance:.3f}m")
    print(
        f"[navigation] GOAL_REACHED_CONFIRMATIONS={args.goal_reached_confirmations}"
    )
    print(
        "[safety] NAVIGATION_SAFETY_PROFILE "
        f"vx=[0,{args.navigation_vx_max:.6f}] "
        f"vy=[-{args.navigation_vy_max:.6f},+{args.navigation_vy_max:.6f}] "
        "wz=unlimited"
    )
    print(f"[safety] LOWSTATE_MAX_AGE={args.max_sensor_age:.3f}s")

    mailbox = NavigationMailbox(args.navigation_command_max_age)
    config = {
        "sensor_socket": args.sensor_socket,
        "navigation_policy": str(Path(args.navigation_policy).expanduser().resolve()),
        "navigation_metadata": str(Path(args.navigation_metadata).expanduser().resolve()),
        "navigation_hz": args.navigation_hz,
        "navigation_command_max_age": args.navigation_command_max_age,
        "command_filter_alpha": args.navigation_filter_alpha,
        "navigation_vx_max": args.navigation_vx_max,
        "navigation_vy_max": args.navigation_vy_max,
        "connect_timeout": args.navigation_connect_timeout,
        "summary_interval": args.navigation_summary_interval,
        "navigation_log": args.navigation_log,
        "goal_tolerance": args.goal_tolerance,
        "goal_reached_confirmations": args.goal_reached_confirmations,
        "lowstate_max_age": args.max_sensor_age,
        "odom_max_age": args.odom_max_age,
        "lidar_max_age": 0.20,
        "assume_clear_lidar": args.assume_clear_lidar,
    }
    navigation = NavigationProcess(config, mailbox)
    controller = SeaNavHimLocoController(args, navigation)
    try:
        controller.validate_model()
        if controller.policy_profile != "himloco_1460" and not args.allow_policy_override:
            raise SafetyError("SEA-Nav navigation requires policy_profile=himloco_1460")
        controller.args.command = FixedCommand(0.0, 0.0, 0.0)
        controller.connect()
        navigation.start()
        controller.run()
    except KeyboardInterrupt:
        print("[safety] Ctrl+C received")
        controller.stop()
    except Exception as exc:
        print(f"[safety] refusing/stopping: {exc}")
        controller.stop()
        raise


if __name__ == "__main__":
    main()

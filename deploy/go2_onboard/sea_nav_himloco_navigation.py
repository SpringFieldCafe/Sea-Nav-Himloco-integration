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
    FixedCommand,
    FixedHIMLocoController,
    SafetyError,
    configure_torch_runtime,
)
from .ipc_schema import decode_packet
from .model_loader import infer, load_navigation_policy
from .navigation_observation import NavigationObservation


DEFAULT_NAVIGATION_POLICY = "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt"
DEFAULT_NAVIGATION_METADATA = "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json"
DEFAULT_SENSOR_SOCKET = "/tmp/sea_nav_shadow.sock"
DEFAULT_NAVIGATION_LOG = "logs/go2_navigation/navigation.jsonl"
SEA_SHA256 = "d1242c74ff56189f20d7a12948d86078287651cd1f70215d9c308d04f4b561de"
HIM_1460_SHA256 = "cab2489dda7732a7d6f51595aa6362384445c738537d0c1c91569054c7b9f5d1"
NAV_LOWER = np.asarray([0.0, 0.0, -0.15], dtype=np.float32)
NAV_UPPER = np.asarray([0.15, 0.0, 0.15], dtype=np.float32)


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


class NavigationLimiter:
    """Explicit first-milestone limiter: forward arc only, no lateral vy."""

    def __init__(self, filter_alpha: float = 0.15):
        self.bridge = ReadOnlyCommandBridge(NAV_LOWER, NAV_UPPER, filter_alpha)

    def reset(self):
        self.bridge.reset()

    def apply(self, raw: Sequence[float]):
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise FloatingPointError("SEA-Nav command must be a finite 3-vector")
        limited = np.clip(raw, NAV_LOWER, NAV_UPPER)
        limited[1] = 0.0
        safe = self.bridge.filter(limited)
        safe[1] = 0.0
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


def _navigation_result(packet, sea, nav_obs, limiter, previous_command, args, reached):
    fresh, reason = _packet_is_fresh(packet, {
        "lowstate": args.lowstate_max_age,
        "odom": args.odom_max_age,
        "lidar": args.lidar_max_age,
    })
    goal_body = np.asarray(packet.get("goal_body", [0.0, 0.0]), dtype=np.float32)
    distance = float(np.linalg.norm(goal_body))
    if reached or (fresh and goal_is_reached(goal_body, args.goal_tolerance)):
        return {
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
        }, True
    if not fresh:
        nav_obs.reset()
        limiter.reset()
        return {
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
        }, reached

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
    observation = nav_obs.build(
        packet_tensor["projected_gravity"],
        command_tensor,
        packet_tensor["base_linear_velocity_body"],
        packet_tensor["base_angular_velocity_body"],
        packet_tensor["lidar_rays"],
        packet_tensor["goal_body"],
    )
    inference_start = time.perf_counter()
    raw = infer(sea, observation, 3)[0].detach().cpu().numpy()
    inference_ms = (time.perf_counter() - inference_start) * 1000.0
    raw_command, limited, safe = limiter.apply(raw)
    return {
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
    }, reached


def _navigation_worker_main(config, sender, stop_event):
    """Run SEA inference away from the 50 Hz LowCmd process."""
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
        limiter = NavigationLimiter(config["command_filter_alpha"])
        previous_command = np.zeros(3, dtype=np.float32)
        reached = False
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(float(config["connect_timeout"]))
        sock.connect(config["sensor_socket"])
        sock.settimeout(0.2)
        buffer = b""
        latest_packet = None
        next_inference = 0.0
        log = None
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
            result, reached = _navigation_result(
                latest_packet, sea, nav_obs, limiter, previous_command, config, reached
            )
            previous_command = np.asarray(result["command"], dtype=np.float32)
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
        sock.close()
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
                "timestamp_monotonic": time.monotonic(),
                "lowcmd_sent": False,
            })
        except (BrokenPipeError, EOFError, OSError):
            pass


class NavigationMailbox:
    """Thread-safe latest-command mailbox with a fail-closed stale result."""

    def __init__(self, max_age: float):
        self.max_age = float(max_age)
        self.lock = threading.RLock()
        self.command = np.zeros(3, dtype=np.float32)
        self.received_at = 0.0
        self.last_result = {"runtime_state": "WAITING", "fault_reason": "navigation_not_ready"}

    def update(self, result):
        command = np.asarray(result.get("command", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        if command.shape != (3,) or not np.isfinite(command).all():
            command = np.zeros(3, dtype=np.float32)
            result = dict(result, runtime_state="NAVIGATION_FAILED", fault_reason="invalid_command")
        with self.lock:
            self.command = command
            self.received_at = time.monotonic()
            self.last_result = dict(result)

    def current(self):
        with self.lock:
            age = time.monotonic() - self.received_at if self.received_at else float("inf")
            if age > self.max_age:
                return FixedCommand(0.0, 0.0, 0.0), "navigation_command_stale", age
            return FixedCommand(*self.command.tolist()), "", age


class NavigationProcess:
    def __init__(self, config, mailbox):
        self.config = dict(config)
        self.mailbox = mailbox
        self.context = mp.get_context("spawn")
        self.stop_event = self.context.Event()
        self.receiver, self.sender = self.context.Pipe(duplex=False)
        self.process = None
        self.receiver_thread = None

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
                    self.mailbox.update(self.receiver.recv())
                elif self.process is not None and not self.process.is_alive():
                    return
            except (EOFError, OSError):
                return

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

    def _run_policy_once(self, _command, snapshot):
        return super()._run_policy_once(self._current_navigation_command(), snapshot)

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
    parser.add_argument("--navigation-connect-timeout", type=float, default=10.0)
    parser.add_argument("--navigation-summary-interval", type=float, default=1.0)
    parser.add_argument("--navigation-log", default=DEFAULT_NAVIGATION_LOG)
    parser.add_argument("--goal-x", type=float, required=True)
    parser.add_argument("--goal-y", type=float, required=True)
    parser.add_argument("--goal-tolerance", type=float, default=0.30)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.legacy_repro or args.comm_only:
        raise SystemExit("SEA-Nav navigation only supports the HIMLoco 1460 active profile")
    if not 0.0 < args.navigation_hz <= 20.0:
        raise SystemExit("--navigation-hz must be in (0,20]")
    if args.navigation_command_max_age <= 0.0 or args.goal_tolerance <= 0.0:
        raise SystemExit("navigation freshness and goal tolerance must be positive")
    # The fixed controller parser is reused for its transport/safety options.
    # Navigation owns the command source; non-zero fixed commands are rejected
    # instead of silently becoming a second command path.
    if any(abs(float(value)) > 1e-8 for value in (args.vx, args.vy, args.wz)):
        raise SystemExit("navigation entry requires fixed command placeholders --vx 0 --vy 0 --wz 0")
    configure_torch_runtime(args.torch_threads, args.torch_interop_threads)
    sea = validate_navigation_model(args.navigation_policy, args.navigation_metadata)
    print(f"[model] SEA path={sea.path} sha256={sea.sha256} input=550 output=3")
    if sha256_file(str(Path(args.policy).expanduser().resolve())) != HIM_1460_SHA256:
        raise SystemExit("SEA-Nav navigation entry requires the approved HIMLoco 1460 model")
    print(f"[navigation] GOAL_WORLD=[{args.goal_x:.6f},{args.goal_y:.6f}] frame=odom")
    print(f"[navigation] GOAL_TOLERANCE={args.goal_tolerance:.3f}m")
    print("[safety] NAVIGATION_SAFETY_PROFILE vx=[0,0.15] vy=0 wz=[-0.15,+0.15]")

    mailbox = NavigationMailbox(args.navigation_command_max_age)
    config = {
        "sensor_socket": args.sensor_socket,
        "navigation_policy": str(Path(args.navigation_policy).expanduser().resolve()),
        "navigation_metadata": str(Path(args.navigation_metadata).expanduser().resolve()),
        "navigation_hz": args.navigation_hz,
        "navigation_command_max_age": args.navigation_command_max_age,
        "command_filter_alpha": args.navigation_filter_alpha,
        "connect_timeout": args.navigation_connect_timeout,
        "summary_interval": args.navigation_summary_interval,
        "navigation_log": args.navigation_log,
        "goal_tolerance": args.goal_tolerance,
        "lowstate_max_age": args.max_sensor_age,
        "odom_max_age": 0.10,
        "lidar_max_age": 0.20,
    }
    navigation = NavigationProcess(config, mailbox)
    controller = SeaNavHimLocoController(args, navigation)
    try:
        controller.validate_model()
        if controller.policy_profile != "himloco_1460":
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

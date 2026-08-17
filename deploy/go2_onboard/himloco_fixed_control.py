"""Conservative HIMLoco-only Go2 low-level controller.

This is intentionally separate from the read-only shadow runtime.  It uses the
Unitree SDK2 ``rt/lowstate``/``rt/lowcmd`` path from the historical Go2
deployment, but refuses to arm unless the current 1460 policy and control-owner
checks pass.  No SEA-Nav or ROS2 control path is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .himloco_observation import HIMLocoObservation
from .joint_mapping import make_policy_to_motor


DEFAULT_POLICY = "models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt"
EXPECTED_SHA256 = "cab2489dda7732a7d6f51595aa6362384445c738537d0c1c91569054c7b9f5d1"
EXPECTED_INPUT_DIM = 270
EXPECTED_OUTPUT_DIM = 12
CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ
POSE_TRANSITION_DURATION = 2.0
POSE_KP = 40.0
POSE_KD = 0.6
STALE_MAX_AGE = 0.10
ARM_WAIT_TIMEOUT = 5.0
ACTION_CLIP = 100.0
ACTION_SCALE = 0.25
POLICY_WARMUP_STEPS = 10
TORCH_THREADS = 1
TORCH_INTEROP_THREADS = 1
KP = 20.0
KD = 0.5
COMMAND_SCALE = np.asarray([2.0, 2.0, 0.25], dtype=np.float32)
DEFAULT_ANGLES = np.asarray(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
     0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    dtype=np.float32,
)

# This is the mapping in the old, known Unitree SDK2 Go2 deployment.
POLICY_TO_MOTOR = tuple(make_policy_to_motor())


class RuntimeState(str, Enum):
    INIT = "INIT"
    WAIT_FOR_POSE_ARM = "WAIT_FOR_POSE_ARM"
    POSE_TRANSITION = "POSE_TRANSITION"
    DEFAULT_POSE_HOLD = "DEFAULT_POSE_HOLD"
    WAIT_FOR_POLICY_ARM = "WAIT_FOR_POLICY_ARM"
    ACTIVE = "ACTIVE"
    STOPPING = "STOPPING"
    DAMPING = "DAMPING"
    EXIT = "EXIT"


class SafetyError(RuntimeError):
    """Fail-closed error before or during real-robot control."""


@dataclass(frozen=True)
class FixedCommand:
    vx: float
    vy: float
    wz: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.vx, self.vy, self.wz], dtype=np.float32)


@dataclass
class LowStateSnapshot:
    message: object = None
    received_at: float = 0.0
    remote_keys: int = 0


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_torch_runtime(threads=TORCH_THREADS, interop_threads=TORCH_INTEROP_THREADS):
    """Use the measured single-thread setting for deterministic 50 Hz control."""
    try:
        torch.set_num_threads(int(threads))
        torch.set_num_interop_threads(int(interop_threads))
    except RuntimeError as exc:
        raise SafetyError(f"unable to configure Torch realtime threads: {exc}") from exc


def validate_fixed_command(vx: float, vy: float, wz: float) -> FixedCommand:
    """Allow only the first three deliberately small test modes."""
    values = np.asarray([vx, vy, wz], dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise SafetyError("fixed command must be three finite numbers")
    if abs(float(vy)) > 1e-8:
        raise SafetyError("first milestone permits vy=0 only")
    if abs(float(vx)) > 0.15 or abs(float(wz)) > 0.15:
        raise SafetyError("fixed command exceeds conservative first-test limit 0.15")
    nonzero = int(np.count_nonzero(np.abs(values) > 1e-8))
    if nonzero > 1 or (abs(float(vx)) > 1e-8 and float(vx) < 0.0):
        raise SafetyError("only zero, tiny forward, or tiny yaw is allowed")
    return FixedCommand(float(vx), float(vy), float(wz))


def build_target_q(action: Sequence[float]) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (12,) or not np.isfinite(action).all():
        raise SafetyError("HIMLoco action must be a finite 12-vector")
    clipped = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
    return DEFAULT_ANGLES + ACTION_SCALE * clipped


def pose_transition_target(initial_q: Sequence[float], step: int, steps: int) -> np.ndarray:
    initial_q = np.asarray(initial_q, dtype=np.float32).reshape(-1)
    if initial_q.shape != (12,) or steps < 2 or not 0 <= step < steps:
        raise SafetyError("invalid pose transition inputs")
    alpha = step / float(steps - 1)
    return initial_q * (1.0 - alpha) + DEFAULT_ANGLES * alpha


def build_observation(
    him_obs: HIMLocoObservation,
    command: Sequence[float],
    gyro_body: Sequence[float],
    projected_gravity: Sequence[float],
    joint_pos_policy: Sequence[float],
    joint_vel_policy: Sequence[float],
    repeat_history: bool = False,
) -> torch.Tensor:
    """Build the exact 45x6 current deployment contract."""
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32).reshape(1, -1)
    command_t = tensor(command)
    gyro_t = tensor(gyro_body)
    gravity_t = tensor(projected_gravity)
    q_t = tensor(joint_pos_policy) - tensor(DEFAULT_ANGLES)
    dq_t = tensor(joint_vel_policy)
    return him_obs.build(command_t, gyro_t, gravity_t, q_t, dq_t, repeat_history=repeat_history)


def decode_remote_keys(raw_remote) -> int:
    """Decode the SDK2 Go2 wireless_remote bit field used by the old entry."""
    raw = bytes(raw_remote)
    if len(raw) < 4:
        return 0
    return struct.unpack("<H", raw[2:4])[0]


def key_pressed(keys: int, index: int) -> bool:
    return bool(keys & (1 << index))


class LowStateWatchdog:
    def __init__(self, max_age: float = STALE_MAX_AGE):
        self.max_age = float(max_age)

    def fresh(self, received_at: float, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        return received_at > 0.0 and 0.0 <= now - received_at <= self.max_age


class FixedHIMLocoController:
    """SDK2 transport and safety state machine; constructed only on the field."""

    ARM_KEY = 8       # A, from historical RemoteController.KeyMap
    POSE_ARM_KEY = 2   # Start, historical zero-torque/pose transition trigger
    STOP_KEY = 3      # Select, historical normal exit
    ESTOP_KEY = 9     # B, independent emergency stop for this milestone

    def __init__(self, args):
        self.args = args
        self.state = RuntimeState.INIT
        self.snapshot = LowStateSnapshot()
        self.snapshot_lock = threading.RLock()
        self.watchdog = LowStateWatchdog(args.max_sensor_age)
        self.sent_any_command = False
        self.policy = None
        self.him_obs = HIMLocoObservation(torch.device("cpu"))
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.low_cmd = None
        self.publisher = None
        self.lowstate_subscriber = None
        self.crc = None
        self._sdk = None
        self.lowstate_rx_count = 0
        self.last_lowstate_rx = 0.0
        self.lowstate_intervals = deque(maxlen=512)
        self.lowstate_callback_durations = deque(maxlen=512)
        self.policy_durations = deque(maxlen=512)
        self.prepare_durations = deque(maxlen=512)
        self.observation_durations = deque(maxlen=512)
        self.forward_durations = deque(maxlen=512)
        self.action_durations = deque(maxlen=512)
        self.publish_durations = deque(maxlen=512)
        self.last_policy_ms = 0.0
        self.last_prepare_ms = 0.0
        self.last_observation_ms = 0.0
        self.last_forward_ms = 0.0
        self.last_action_ms = 0.0
        self.last_publish_ms = 0.0
        self.loop_durations = deque(maxlen=512)
        self.control_periods = deque(maxlen=512)
        self.deadline_lateness = deque(maxlen=512)
        self.deadline_miss_count = 0
        self._last_summary = 0.0

    def validate_model(self):
        path = Path(self.args.policy).resolve()
        if not path.is_file():
            raise SafetyError(f"HIMLoco model does not exist: {path}")
        actual = sha256_file(str(path))
        if actual != EXPECTED_SHA256:
            raise SafetyError(f"model SHA256 mismatch: expected {EXPECTED_SHA256}, got {actual}")
        self.policy = torch.jit.load(str(path), map_location="cpu").eval()
        with torch.inference_mode():
            probe = torch.zeros((1, EXPECTED_INPUT_DIM), dtype=torch.float32)
            output = self.policy(probe)
        if tuple(output.shape) != (1, EXPECTED_OUTPUT_DIM):
            raise SafetyError(f"model contract must be 270->12, got {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise SafetyError("model contract probe returned NaN/Inf")
        with torch.inference_mode():
            for _ in range(POLICY_WARMUP_STEPS):
                warmup_output = self.policy(probe)
        if not torch.isfinite(warmup_output).all():
            raise SafetyError("model warm-up returned NaN/Inf")
        print(f"[model] path={path}")
        print(f"[model] sha256={actual} input=270 output=12 warmup={POLICY_WARMUP_STEPS}")

    def check_motion_owner(self):
        """Refuse ARM when Unitree high-level motion service owns the robot."""
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        except ImportError as exc:
            raise SafetyError("MotionSwitcherClient unavailable; refusing to ARM") from exc
        client = MotionSwitcherClient()
        client.SetTimeout(5.0)
        client.Init()
        result = client.CheckMode()
        if isinstance(result, tuple):
            code, data = result
        else:
            code, data = 0, result
        if int(code) != 0:
            raise SafetyError(f"MotionSwitcher CheckMode failed: {code}")
        name = str((data or {}).get("name", "")) if isinstance(data, dict) else str(data or "")
        if name:
            raise SafetyError(
                f"controller conflict: Unitree motion service owns '{name}'. "
                "Release it manually and rerun; this program will not kill services."
            )
        print("[safety] motion owner check: no high-level owner reported")

    def connect(self):
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo, LowState_ as LowStateGo
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as exc:
            raise SafetyError("Unitree SDK2 imports unavailable; cannot start real controller") from exc
        self._sdk = (ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize, unitree_go_msg_dds__LowCmd_, LowCmdGo, LowStateGo, CRC)
        ChannelFactoryInitialize(0, self.args.net)
        self.check_motion_owner()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmdGo)
        self.publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        # Match the previously working deployment: DDS only enqueues samples;
        # the SDK reader thread invokes the Python snapshot callback. This
        # keeps Python callback work out of CycloneDDS's listener thread.
        self.lowstate_subscriber.Init(self._low_state_callback, 10)
        self._initialize_command(self.low_cmd)
        print("[safety] SDK2 connected; no policy command will be sent before Start/A sequence")

    def _initialize_command(self, command):
        command.head[0] = 0xFE
        command.head[1] = 0xEF
        command.level_flag = 0xFF
        command.gpio = 0
        for motor in command.motor_cmd:
            motor.mode = 0x0A
            motor.q = 2.146e9
            motor.dq = 16000.0
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = 0.0

    def _low_state_callback(self, message):
        callback_start = time.monotonic()
        received_at = time.monotonic()
        try:
            keys = decode_remote_keys(message.wireless_remote)
        except (AttributeError, TypeError, ValueError):
            keys = 0
        with self.snapshot_lock:
            if self.last_lowstate_rx > 0.0:
                self.lowstate_intervals.append(received_at - self.last_lowstate_rx)
            self.last_lowstate_rx = received_at
            self.lowstate_rx_count += 1
            self.snapshot = LowStateSnapshot(message, received_at, keys)
        self.lowstate_callback_durations.append(time.monotonic() - callback_start)

    def _snapshot(self):
        with self.snapshot_lock:
            return self.snapshot

    def _send(self, command):
        if self.publisher is None:
            raise SafetyError("publisher is not initialized")
        command.crc = self._sdk[-1]().Crc(command)
        self.publisher.Write(command)
        self.sent_any_command = True

    def _send_damping(self, frames=10):
        if not self.sent_any_command or self.low_cmd is None:
            return
        for motor in self.low_cmd.motor_cmd:
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 8.0
            motor.tau = 0.0
        for _ in range(frames):
            self._send(self.low_cmd)
            time.sleep(CONTROL_DT)

    def _prepare_sensor(self, message):
        motors = getattr(message, "motor_state", None)
        if motors is None or len(motors) < 12:
            raise SafetyError("LowState has fewer than 12 motors")
        imu = message.imu_state
        gyro = np.asarray(imu.gyroscope, dtype=np.float32)
        quat = np.asarray(imu.quaternion, dtype=np.float32)
        q_motor = np.asarray([motor.q for motor in motors[:12]], dtype=np.float32)
        dq_motor = np.asarray([motor.dq for motor in motors[:12]], dtype=np.float32)
        if not all(np.isfinite(value).all() for value in (gyro, quat, q_motor, dq_motor)):
            raise SafetyError("LowState contains NaN/Inf")
        if quat.shape != (4,) or np.linalg.norm(quat) < 1e-6:
            raise SafetyError("invalid IMU quaternion")
        q_policy = q_motor[list(POLICY_TO_MOTOR)]
        dq_policy = dq_motor[list(POLICY_TO_MOTOR)]
        gravity = projected_gravity_from_wxyz(quat)
        return gyro, gravity, q_policy, dq_policy

    def _run_policy_once(self, command, snapshot):
        prepare_start = time.monotonic()
        gyro, gravity, q_policy, dq_policy = self._prepare_sensor(snapshot.message)
        self.last_prepare_ms = (time.monotonic() - prepare_start) * 1000.0
        self.prepare_durations.append(time.monotonic() - prepare_start)
        observation_start = time.monotonic()
        observation = build_observation(self.him_obs, command.as_array(), gyro, gravity, q_policy, dq_policy)
        self.last_observation_ms = (time.monotonic() - observation_start) * 1000.0
        self.observation_durations.append(time.monotonic() - observation_start)
        forward_start = time.monotonic()
        with torch.inference_mode():
            action = self.policy(observation).reshape(-1).detach().cpu().numpy()
        self.last_forward_ms = (time.monotonic() - forward_start) * 1000.0
        self.forward_durations.append(time.monotonic() - forward_start)
        action_start = time.monotonic()
        target_policy = build_target_q(action)
        for policy_index, motor_index in enumerate(POLICY_TO_MOTOR):
            motor = self.low_cmd.motor_cmd[motor_index]
            motor.q = float(target_policy[policy_index])
            motor.dq = 0.0
            motor.kp = KP
            motor.kd = KD
            motor.tau = 0.0
        self.him_obs.record_action(torch.from_numpy(np.clip(action, -ACTION_CLIP, ACTION_CLIP)).reshape(1, 12))
        self.last_action_ms = (time.monotonic() - action_start) * 1000.0
        self.action_durations.append(time.monotonic() - action_start)
        return action

    def _check_runtime_safety(self, snapshot):
        if not self.watchdog.fresh(snapshot.received_at):
            raise SafetyError(f"LowState stale during {self.state.value}")
        if key_pressed(snapshot.remote_keys, self.ESTOP_KEY):
            raise SafetyError("wireless ESTOP")
        if key_pressed(snapshot.remote_keys, self.STOP_KEY):
            raise SafetyError("wireless STOP")

    def _set_pose_command(self, target_policy, kp, kd):
        target_policy = np.asarray(target_policy, dtype=np.float32).reshape(-1)
        if target_policy.shape != (12,) or not np.isfinite(target_policy).all():
            raise SafetyError("pose target must be a finite 12-vector")
        for policy_index, motor_index in enumerate(POLICY_TO_MOTOR):
            motor = self.low_cmd.motor_cmd[motor_index]
            motor.q = float(target_policy[policy_index])
            motor.dq = 0.0
            motor.kp = float(kp)
            motor.kd = float(kd)
            motor.tau = 0.0

    def _set_zero_torque_command(self):
        for motor in self.low_cmd.motor_cmd:
            motor.q = 2.146e9
            motor.dq = 16000.0
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = 0.0

    def _send_timed_pose_command(self, target_policy, kp, kd):
        start = time.monotonic()
        self._set_pose_command(target_policy, kp, kd)
        publish_start = time.monotonic()
        self._send(self.low_cmd)
        self.last_publish_ms = (time.monotonic() - publish_start) * 1000.0
        self.loop_durations.append(time.monotonic() - start)

    def _wait_for_pose_arm(self):
        self.state = RuntimeState.WAIT_FOR_POSE_ARM
        print("[safety] WAIT_FOR_POSE_ARM: press Start to begin pose transition; Select=STOP; B=ESTOP; Ctrl+C=STOP")
        wait_deadline = time.monotonic() + ARM_WAIT_TIMEOUT
        while True:
            snapshot = self._snapshot()
            if not self.watchdog.fresh(snapshot.received_at):
                if time.monotonic() >= wait_deadline:
                    raise SafetyError(f"no fresh LowState received within {ARM_WAIT_TIMEOUT:.1f}s before pose transition")
                time.sleep(CONTROL_DT)
                continue
            self._check_runtime_safety(snapshot)
            if key_pressed(snapshot.remote_keys, self.POSE_ARM_KEY):
                return snapshot
            self._set_zero_torque_command()
            self._send(self.low_cmd)
            time.sleep(CONTROL_DT)

    def _move_to_default_pos(self, initial_q):
        self.state = RuntimeState.POSE_TRANSITION
        initial_q = np.asarray(initial_q, dtype=np.float32).reshape(12)
        max_error = float(np.max(np.abs(initial_q - DEFAULT_ANGLES)))
        print(f"[safety] POSE_TRANSITION_START max_initial_joint_error={max_error:.6f}")
        steps = max(2, int(round(POSE_TRANSITION_DURATION / CONTROL_DT)))
        next_tick = time.monotonic()
        for step in range(steps):
            snapshot = self._snapshot()
            self._check_runtime_safety(snapshot)
            target = pose_transition_target(initial_q, step, steps)
            self._send_timed_pose_command(target, POSE_KP, POSE_KD)
            next_tick += CONTROL_DT
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
        print("[safety] POSE_TRANSITION_COMPLETE")

    def _wait_for_policy_arm(self):
        self.state = RuntimeState.DEFAULT_POSE_HOLD
        print("[safety] DEFAULT_POSE_HOLD: press A for policy; Select=STOP; B=ESTOP; Ctrl+C=STOP")
        released = False
        next_tick = time.monotonic()
        while True:
            snapshot = self._snapshot()
            self._check_runtime_safety(snapshot)
            self._send_timed_pose_command(DEFAULT_ANGLES, POSE_KP, POSE_KD)
            self.state = RuntimeState.WAIT_FOR_POLICY_ARM
            if not key_pressed(snapshot.remote_keys, self.ARM_KEY):
                released = True
            elif released:
                return snapshot
            next_tick += CONTROL_DT
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()

    def _initialize_policy_history(self, snapshot, command):
        gyro, gravity, q_policy, dq_policy = self._prepare_sensor(snapshot.message)
        build_observation(
            self.him_obs,
            command.as_array(),
            gyro,
            gravity,
            q_policy,
            dq_policy,
            repeat_history=True,
        )
        print("[safety] HISTORY_INIT_SOURCE=latest_default_pose_lowstate PREVIOUS_ACTION_INIT=zeros")

    def _print_diagnostics(self, now):
        if now - self._last_summary < 1.0:
            return
        self._last_summary = now
        with self.snapshot_lock:
            rx_count = self.lowstate_rx_count
            intervals = np.asarray(self.lowstate_intervals, dtype=np.float64)
            age_ms = (now - self.snapshot.received_at) * 1000.0
        if intervals.size:
            rx_rate = 1.0 / float(np.median(intervals))
            p50 = float(np.percentile(intervals, 50) * 1000.0)
            p95 = float(np.percentile(intervals, 95) * 1000.0)
        else:
            rx_rate, p50, p95 = 0.0, 0.0, 0.0
        loops = np.asarray(self.loop_durations, dtype=np.float64)
        callbacks = np.asarray(self.lowstate_callback_durations, dtype=np.float64)
        policies = np.asarray(self.policy_durations, dtype=np.float64)
        prepares = np.asarray(self.prepare_durations, dtype=np.float64)
        observations = np.asarray(self.observation_durations, dtype=np.float64)
        forwards = np.asarray(self.forward_durations, dtype=np.float64)
        actions = np.asarray(self.action_durations, dtype=np.float64)
        publishes = np.asarray(self.publish_durations, dtype=np.float64)
        periods = np.asarray(self.control_periods, dtype=np.float64)
        lateness = np.asarray(self.deadline_lateness, dtype=np.float64)
        loop_p95 = float(np.percentile(loops, 95) * 1000.0) if loops.size else 0.0
        callback_p95 = float(np.percentile(callbacks, 95) * 1000.0) if callbacks.size else 0.0
        policy_p95 = float(np.percentile(policies, 95) * 1000.0) if policies.size else 0.0
        prepare_p95 = float(np.percentile(prepares, 95) * 1000.0) if prepares.size else 0.0
        observation_p95 = float(np.percentile(observations, 95) * 1000.0) if observations.size else 0.0
        forward_p95 = float(np.percentile(forwards, 95) * 1000.0) if forwards.size else 0.0
        action_p95 = float(np.percentile(actions, 95) * 1000.0) if actions.size else 0.0
        publish_p95 = float(np.percentile(publishes, 95) * 1000.0) if publishes.size else 0.0
        period_values = periods * 1000.0
        period_p50 = float(np.percentile(period_values, 50)) if periods.size else 0.0
        period_p95 = float(np.percentile(period_values, 95)) if periods.size else 0.0
        period_p99 = float(np.percentile(period_values, 99)) if periods.size else 0.0
        period_max = float(np.max(period_values)) if periods.size else 0.0
        rate = 1000.0 / period_p50 if period_p50 > 0.0 else 0.0
        late_max = float(np.max(lateness) * 1000.0) if lateness.size else 0.0
        print(
            "[diagnostics] state=%s lowstate_rx=%d lowstate_rate_hz=%.1f "
            "rx_p50_ms=%.2f rx_p95_ms=%.2f age_ms=%.2f policy_ms=%.2f "
            "policy_p95_ms=%.2f publish_ms=%.2f publish_p95_ms=%.2f "
            "callback_p95_ms=%.2f prepare_p95_ms=%.2f obs_p95_ms=%.2f "
            "forward_p95_ms=%.2f action_p95_ms=%.2f "
            "loop_p95_ms=%.2f control_rate_hz=%.2f period_p50_ms=%.2f "
            "period_p95_ms=%.2f period_p99_ms=%.2f period_max_ms=%.2f "
            "deadline_misses=%d max_deadline_lateness_ms=%.2f"
            % (self.state.value, rx_count, rx_rate, p50, p95, age_ms,
               self.last_policy_ms, policy_p95, self.last_publish_ms,
               publish_p95, callback_p95, prepare_p95, observation_p95,
               forward_p95, action_p95, loop_p95, rate, period_p50, period_p95,
               period_p99, period_max, self.deadline_miss_count, late_max)
        )

    def _reset_active_metrics(self):
        self.lowstate_callback_durations.clear()
        self.policy_durations.clear()
        self.prepare_durations.clear()
        self.observation_durations.clear()
        self.forward_durations.clear()
        self.action_durations.clear()
        self.publish_durations.clear()
        self.loop_durations.clear()
        self.control_periods.clear()
        self.deadline_lateness.clear()
        self.deadline_miss_count = 0

    def run(self):
        pose_snapshot = self._wait_for_pose_arm()
        _, _, initial_q, _ = self._prepare_sensor(pose_snapshot.message)
        self._move_to_default_pos(initial_q)
        policy_snapshot = self._wait_for_policy_arm()
        self._initialize_policy_history(policy_snapshot, self.args.command)
        self.state = RuntimeState.ACTIVE
        self._reset_active_metrics()
        print(f"[safety] ACTIVE command={self.args.command.as_array().tolist()}")

        next_tick = time.monotonic()
        previous_control_start = None
        while self.state == RuntimeState.ACTIVE:
            loop_start = time.monotonic()
            if previous_control_start is not None:
                self.control_periods.append(loop_start - previous_control_start)
            previous_control_start = loop_start
            next_tick += CONTROL_DT
            snapshot = self._snapshot()
            self._check_runtime_safety(snapshot)
            policy_start = time.monotonic()
            self._run_policy_once(self.args.command, snapshot)
            self.last_policy_ms = (time.monotonic() - policy_start) * 1000.0
            self.policy_durations.append(time.monotonic() - policy_start)
            publish_start = time.monotonic()
            self._send(self.low_cmd)
            self.last_publish_ms = (time.monotonic() - publish_start) * 1000.0
            self.publish_durations.append(time.monotonic() - publish_start)
            self.loop_durations.append(time.monotonic() - loop_start)
            lateness = max(0.0, time.monotonic() - next_tick)
            self.deadline_lateness.append(lateness)
            if lateness > 0.0:
                self.deadline_miss_count += 1
            self._print_diagnostics(time.monotonic())
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
        self.stop()

    def stop(self):
        self.state = RuntimeState.STOPPING
        try:
            self.state = RuntimeState.DAMPING
            self._send_damping()
        finally:
            self.state = RuntimeState.EXIT
            print("[safety] EXIT lowcmd_sent=%s" % self.sent_any_command)


def projected_gravity_from_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    """Return gravity projected into body coordinates for w,x,y,z IMU data."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float32).reshape(4)
    return np.asarray([
        2.0 * (w * y - x * z),
        -2.0 * (w * x + y * z),
        -1.0 + 2.0 * (x * x + y * y),
    ], dtype=np.float32)


def build_parser():
    parser = argparse.ArgumentParser(description="HIMLoco 1460 fixed-command Go2 controller")
    parser.add_argument("net", help="Ethernet interface, e.g. enp3s0")
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--vx", type=float, required=True)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--max-sensor-age", type=float, default=STALE_MAX_AGE)
    parser.add_argument("--torch-threads", type=int, choices=(1, 2, 4), default=TORCH_THREADS)
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        choices=(1, 2, 4),
        default=TORCH_INTEROP_THREADS,
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_torch_runtime(args.torch_threads, args.torch_interop_threads)
    args.command = validate_fixed_command(args.vx, args.vy, args.wz)
    controller = FixedHIMLocoController(args)
    try:
        controller.validate_model()
        controller.connect()
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

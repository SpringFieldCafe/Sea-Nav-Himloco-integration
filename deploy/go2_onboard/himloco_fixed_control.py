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
STALE_MAX_AGE = 0.10
ACTION_CLIP = 100.0
ACTION_SCALE = 0.25
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
    WAIT_FOR_ARM = "WAIT_FOR_ARM"
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


def build_observation(
    him_obs: HIMLocoObservation,
    command: Sequence[float],
    gyro_body: Sequence[float],
    projected_gravity: Sequence[float],
    joint_pos_policy: Sequence[float],
    joint_vel_policy: Sequence[float],
) -> torch.Tensor:
    """Build the exact 45x6 current deployment contract."""
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32).reshape(1, -1)
    command_t = tensor(command)
    gyro_t = tensor(gyro_body)
    gravity_t = tensor(projected_gravity)
    q_t = tensor(joint_pos_policy) - tensor(DEFAULT_ANGLES)
    dq_t = tensor(joint_vel_policy)
    return him_obs.build(command_t, gyro_t, gravity_t, q_t, dq_t)


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
        self.crc = None
        self._sdk = None

    def validate_model(self):
        path = Path(self.args.policy).resolve()
        if not path.is_file():
            raise SafetyError(f"HIMLoco model does not exist: {path}")
        actual = sha256_file(str(path))
        if actual != EXPECTED_SHA256:
            raise SafetyError(f"model SHA256 mismatch: expected {EXPECTED_SHA256}, got {actual}")
        self.policy = torch.jit.load(str(path), map_location="cpu").eval()
        with torch.inference_mode():
            output = self.policy(torch.zeros((1, EXPECTED_INPUT_DIM), dtype=torch.float32))
        if tuple(output.shape) != (1, EXPECTED_OUTPUT_DIM):
            raise SafetyError(f"model contract must be 270->12, got {tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise SafetyError("model contract probe returned NaN/Inf")
        print(f"[model] path={path}")
        print(f"[model] sha256={actual} input=270 output=12")

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
        subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        subscriber.Init(self._low_state_callback, 10)
        self._initialize_command(self.low_cmd)
        print("[safety] SDK2 connected; no command will be sent before A-arm")

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
        try:
            keys = decode_remote_keys(message.wireless_remote)
        except (AttributeError, TypeError, ValueError):
            keys = 0
        with self.snapshot_lock:
            self.snapshot = LowStateSnapshot(message, time.monotonic(), keys)

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
        gyro, gravity, q_policy, dq_policy = self._prepare_sensor(snapshot.message)
        observation = build_observation(self.him_obs, command.as_array(), gyro, gravity, q_policy, dq_policy)
        with torch.inference_mode():
            action = self.policy(observation).reshape(-1).detach().cpu().numpy()
        target_policy = build_target_q(action)
        for policy_index, motor_index in enumerate(POLICY_TO_MOTOR):
            motor = self.low_cmd.motor_cmd[motor_index]
            motor.q = float(target_policy[policy_index])
            motor.dq = 0.0
            motor.kp = KP
            motor.kd = KD
            motor.tau = 0.0
        self.him_obs.record_action(torch.from_numpy(np.clip(action, -ACTION_CLIP, ACTION_CLIP)).reshape(1, 12))
        return action

    def run(self):
        self.state = RuntimeState.WAIT_FOR_ARM
        print("[safety] WAIT_FOR_ARM: press A to arm; Select=STOP; B=ESTOP; Ctrl+C=STOP")
        while True:
            snapshot = self._snapshot()
            if not self.watchdog.fresh(snapshot.received_at):
                raise SafetyError("LowState stale before ARM")
            if key_pressed(snapshot.remote_keys, self.ESTOP_KEY):
                raise SafetyError("wireless ESTOP before ARM")
            if key_pressed(snapshot.remote_keys, self.ARM_KEY):
                self.state = RuntimeState.ACTIVE
                print(f"[safety] ACTIVE command={self.args.command.as_array().tolist()}")
                break
            time.sleep(0.02)

        next_tick = time.monotonic()
        while self.state == RuntimeState.ACTIVE:
            next_tick += CONTROL_DT
            snapshot = self._snapshot()
            if not self.watchdog.fresh(snapshot.received_at):
                raise SafetyError("LowState stale during ACTIVE")
            if key_pressed(snapshot.remote_keys, self.ESTOP_KEY):
                raise SafetyError("wireless ESTOP")
            if key_pressed(snapshot.remote_keys, self.STOP_KEY):
                self.state = RuntimeState.STOPPING
                break
            self._run_policy_once(self.args.command, snapshot)
            self._send(self.low_cmd)
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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
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

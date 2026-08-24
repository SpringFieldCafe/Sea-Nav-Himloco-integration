"""Original SEA-Nav SLR locomotion backend for Go2.

The navigation policy produces ``vx/vy/wz``.  This module reproduces the
repository's original ``SLRBackend.compute_actions`` contract with the three
tracked TorchScript modules and uses the existing guarded 50 Hz LowCmd state
machine only for the final joint-level transport.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .himloco_fixed_control import (
    DEFAULT_ANGLES,
    FixedCommand,
    FixedHIMLocoController,
    SafetyError,
    configure_torch_runtime,
    POLICY_TO_MOTOR,
)
from .sea_nav_himloco_navigation import (
    DEFAULT_NAVIGATION_METADATA,
    DEFAULT_NAVIGATION_POLICY,
    NavigationMailbox,
    NavigationProcess,
    validate_navigation_model,
)


SLR_HISTORY = 10
SLR_FRAME = 45
SLR_COMMAND_SCALE = np.asarray([2.0, 2.0, 0.25], dtype=np.float32)
SLR_ANGULAR_SCALE = 0.25
SLR_DOF_POSITION_SCALE = 1.0
SLR_DOF_VELOCITY_SCALE = 0.05
SLR_ACTION_SCALE = 0.25
SLR_KP = 30.0
SLR_KD = 0.75
SLR_ACTION_ABS_LIMIT = 6.0


class SeaNavSLRController(FixedHIMLocoController):
    """Reuse transport/safety gates while replacing only policy inference."""

    def __init__(self, args, navigation):
        super().__init__(args)
        self.navigation = navigation
        self.body = None
        self.encoder_vel = None
        self.encoder_latent = None
        self.slr_history = torch.zeros(1, SLR_HISTORY, SLR_FRAME, dtype=torch.float32)
        self.previous_slr_action = np.zeros(12, dtype=np.float32)
        self.last_navigation_command = np.zeros(3, dtype=np.float32)
        self.last_slr_action = np.zeros(12, dtype=np.float32)
        self._last_slr_summary = 0.0
        self._last_slr_input_summary = 0.0

    def validate_model(self):
        root = Path(__file__).resolve().parents[2] / "training/legged_gym/legged_gym/ctrl_model"
        paths = [root / name for name in ("body_latest.jit", "encoder_vel.jit", "encoder_latent.jit")]
        if not all(path.is_file() for path in paths):
            raise SafetyError(f"missing SLR control modules under {root}")
        self.body = torch.jit.load(str(paths[0]), map_location="cpu").eval()
        self.encoder_vel = torch.jit.load(str(paths[1]), map_location="cpu").eval()
        self.encoder_latent = torch.jit.load(str(paths[2]), map_location="cpu").eval()
        with torch.inference_mode():
            probe = torch.zeros((1, SLR_HISTORY * SLR_FRAME))
            vel = self.encoder_vel(probe)
            latent = self.encoder_latent(probe)
            output = self.body(torch.zeros((1, 2 + SLR_FRAME + 1 + 16)))
        if tuple(vel.shape) != (1, 2) or tuple(latent.shape) != (1, 16) or tuple(output.shape) != (1, 12):
            raise SafetyError("SLR TorchScript contract is not 450->2, 450->16, 64->12")
        self.policy_profile = "sea_nav_slr_original"
        print(f"[model] SLR body={paths[0]}")
        print(f"[model] SLR encoders={paths[1]},{paths[2]}")
        print("[model] SLR contract frame=45 history=10 action=12")

    def _navigation_command(self):
        if self.args.zero_navigation_command:
            return FixedCommand(0.0, 0.0, 0.0)
        command, reason, age = self.navigation.mailbox.current()
        if reason:
            raise SafetyError(f"navigation command stopped: {reason} age={age:.3f}s")
        self.last_navigation_command = command.as_array().copy()
        return command

    def _build_slr_frame(self, command, snapshot):
        gyro, gravity, q_policy, dq_policy = self._prepare_sensor(snapshot.message)
        self.last_slr_sensor = (gyro, gravity, q_policy, dq_policy)
        q_offset = q_policy - DEFAULT_ANGLES
        return np.concatenate((
            gyro * SLR_ANGULAR_SCALE,
            gravity,
            command.as_array() * SLR_COMMAND_SCALE,
            q_offset * SLR_DOF_POSITION_SCALE,
            dq_policy * SLR_DOF_VELOCITY_SCALE,
            self.previous_slr_action,
        )).astype(np.float32)

    def _initialize_policy_history(self, snapshot, _command):
        self.navigation.mailbox.wait_for_fresh_command(5.0)
        command = self._navigation_command()
        frame = self._build_slr_frame(command, snapshot)
        self.slr_history[:] = torch.from_numpy(frame).reshape(1, 1, SLR_FRAME)
        self.slr_history[:] = self.slr_history[:, :1].repeat(1, SLR_HISTORY, 1)
        print("[safety] SLR_HISTORY_INIT=repeat_current frame=45 history=10")
        return self.slr_history.reshape(1, -1)

    def _run_policy_once(self, _command, snapshot, observation=None):
        command = self._navigation_command()
        # The parent controller already initializes the repeated history before
        # the first active cycle. Keep that history for the first inference;
        # append one fresh frame only on subsequent cycles.
        if observation is None:
            frame = self._build_slr_frame(command, snapshot)
            self.slr_history = torch.cat(
                (self.slr_history[:, 1:], torch.from_numpy(frame).reshape(1, 1, SLR_FRAME)), dim=1
            )
        slr_input = self.slr_history.reshape(1, SLR_HISTORY * SLR_FRAME)
        with torch.inference_mode():
            velocity = self.encoder_vel(slr_input)
            latent = self.encoder_latent(slr_input)
            prop = slr_input[:, -SLR_FRAME:]
            # The SLR actor receives base angular z again after the 45D frame.
            ang_vel_z = prop[:, 2:3]
            actor_input = torch.cat((velocity, prop, ang_vel_z, latent), dim=1)
            action = self.body(actor_input).reshape(-1).cpu().numpy()
        if action.shape != (12,) or not np.isfinite(action).all():
            raise SafetyError("SLR action is not a finite 12-vector")
        action_abs_max = float(np.max(np.abs(action)))
        if action_abs_max > SLR_ACTION_ABS_LIMIT:
            raise SafetyError(
                "SLR action guard tripped: "
                f"max_abs={action_abs_max:.3f} limit={SLR_ACTION_ABS_LIMIT:.3f}"
            )
        target = DEFAULT_ANGLES + SLR_ACTION_SCALE * action
        for policy_index, motor_index in enumerate(POLICY_TO_MOTOR):
            motor = self.low_cmd.motor_cmd[motor_index]
            motor.q = float(target[policy_index])
            motor.dq = 0.0
            motor.kp = SLR_KP
            motor.kd = SLR_KD
            motor.tau = 0.0
        self.previous_slr_action = action.astype(np.float32)
        self.last_slr_action = action.astype(np.float32)

        now = time.monotonic()
        if now - self._last_slr_input_summary >= 1.0:
            self._last_slr_input_summary = now
            gyro, gravity, q_policy, dq_policy = self.last_slr_sensor
            print(
                "[slr-input] "
                + json.dumps(
                    {
                        "gyro_norm": float(np.linalg.norm(gyro)),
                        "gravity": gravity.tolist(),
                        "q_offset_max": float(np.max(np.abs(q_policy - DEFAULT_ANGLES))),
                        "dq_max": float(np.max(np.abs(dq_policy))),
                        "command": command.as_array().tolist() if hasattr(command, "as_array") else None,
                        "action_norm": float(np.linalg.norm(action)),
                        "action_abs_max": action_abs_max,
                    },
                    separators=(",", ":"),
                )
            )

    def _print_diagnostics(self, now):
        super()._print_diagnostics(now)
        if now - self._last_slr_summary < 1.0:
            return
        self._last_slr_summary = now
        with self.navigation.mailbox.lock:
            result = dict(self.navigation.mailbox.last_result)
        print("[navigation] " + json.dumps({
            "state": result.get("runtime_state"),
            "goal_distance": result.get("goal_distance"),
            "raw": result.get("raw_command"),
            "safe": result.get("command"),
            "slr_action_norm": float(np.linalg.norm(self.last_slr_action)),
            "lowcmd_sent": True,
        }, separators=(",", ":")))

    def stop(self):
        self.navigation.stop()
        super().stop()


def build_parser():
    parser = argparse.ArgumentParser(description="SEA-Nav with original SLR locomotion backend")
    parser.add_argument("net")
    parser.add_argument("--sensor-socket", default="/tmp/sea_nav_slr.sock")
    parser.add_argument("--navigation-policy", default=DEFAULT_NAVIGATION_POLICY)
    parser.add_argument("--navigation-metadata", default=DEFAULT_NAVIGATION_METADATA)
    parser.add_argument("--navigation-log", required=True)
    parser.add_argument("--navigation-vx-max", type=float, default=0.15)
    parser.add_argument("--navigation-vy-max", type=float, default=0.0)
    parser.add_argument("--navigation-filter-alpha", type=float, default=0.15)
    parser.add_argument("--navigation-hz", type=float, default=10.0)
    parser.add_argument("--navigation-command-max-age", type=float, default=0.25)
    parser.add_argument("--navigation-connect-timeout", type=float, default=10.0)
    parser.add_argument("--navigation-summary-interval", type=float, default=1.0)
    parser.add_argument(
        "--zero-navigation-command",
        action="store_true",
        help="ignore SEA-Nav command and feed SLR [0,0,0] for a smoke test",
    )
    parser.add_argument("--goal-tolerance", type=float, default=0.15)
    parser.add_argument("--goal-reached-confirmations", type=int, default=5)
    parser.add_argument("--max-sensor-age", type=float, default=0.10)
    parser.add_argument("--assume-clear-lidar", action="store_true")
    parser.add_argument("--event-trace", default="/tmp/sea_nav_slr_event_trace.jsonl")
    parser.add_argument("--hold-duration", type=float, default=0.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_torch_runtime(args.torch_threads, args.torch_interop_threads)
    validate_navigation_model(args.navigation_policy, args.navigation_metadata)
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
        "odom_max_age": 0.10,
        "lidar_max_age": 0.20,
        "assume_clear_lidar": args.assume_clear_lidar,
    }
    navigation = NavigationProcess(config, mailbox)
    controller = SeaNavSLRController(args, navigation)
    try:
        controller.validate_model()
        args.command = FixedCommand(0.0, 0.0, 0.0)
        controller.connect()
        navigation.start()
        controller.run()
    except KeyboardInterrupt:
        print("[safety] Ctrl+C received")
        controller.stop()
        return 130
    except Exception as exc:
        print(f"[safety] refusing/stopping: {exc}")
        controller.stop()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

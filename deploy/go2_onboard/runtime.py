"""Production-like, read-only Go2 sensor and policy shadow runtime."""

import argparse
import math
import time

import numpy as np

from .command_bridge import ReadOnlyCommandBridge
from .config import RuntimeConfig
from .goal import Goal2D, GoalManager
from .joint_mapping import make_policy_to_motor
from .logger import JsonlLogger
from .ros_state_reader import RosStateReader
from .safety_supervisor import RuntimeState, SafetySupervisor


DEFAULT_JOINT_ANGLES = np.asarray([0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                                   0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float32)


def quat_to_gravity(quaternion):
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.size != 4 or not np.isfinite(q).all():
        raise ValueError("IMU quaternion must be finite [w,x,y,z]")
    w, x, y, z = [float(v) for v in q]
    return np.asarray([2.0 * (w * y - x * z), -2.0 * (w * x + y * z),
                       -1.0 + 2.0 * (x * x + y * y)], dtype=np.float32)


def quat_to_yaw(quaternion):
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Topics:
    def __init__(self, args):
        self.lowstate = args.lowstate_topic
        self.lidar = args.lidar_topic
        self.odom = args.odom_topic
        self.wireless = args.wireless_topic
        self.goal = args.goal_topic


class OnboardRuntime:
    """Shared runtime with an explicit sensor-only or shadow-only mode."""

    def __init__(self, args):
        self.args = args
        self.device = None
        self.config = RuntimeConfig(
            navigation_policy=args.navigation_policy or "",
            navigation_metadata=args.navigation_metadata,
            himloco_policy=args.himloco_policy or "",
            lowstate_topic=args.lowstate_topic,
            lidar_topic=args.lidar_topic,
            odom_topic=args.odom_topic,
            wireless_topic=args.wireless_topic,
            goal_topic=args.goal_topic,
            device=args.device,
            control_hz=args.control_hz,
            max_sensor_age=args.max_sensor_age,
            command_filter_alpha=args.command_filter_alpha,
            goal_xy=[args.goal_x, args.goal_y] if args.goal_x is not None and args.goal_y is not None else None,
            log_interval=args.log_interval,
        )
        self.reader = RosStateReader(Topics(args), max_sensor_age=args.max_sensor_age)
        self.goal_manager = GoalManager()
        self.safety = SafetySupervisor(self.config.max_sensor_age)
        self.logger = JsonlLogger(args.log)
        self.command_bridge = ReadOnlyCommandBridge(
            self.config.command_bounds_min, self.config.command_bounds_max,
            self.config.command_filter_alpha,
        )
        self.rate = 1.0 / args.control_hz
        self.last_command = None
        self.last_report = 0.0
        self.started = time.monotonic()
        self.goal_topic_enabled = bool(args.goal_topic)
        if self.config.goal_xy is not None:
            self.goal_manager.update(Goal2D(args.goal_frame, args.goal_x, args.goal_y, time.time()))
        self.nav_policy = None
        self.him_policy = None
        self.nav_obs = None
        self.him_obs = None
        self.ray_adapter = None
        self.policy_to_motor = make_policy_to_motor()
        if args.mode == "shadow":
            self._load_policies()

    def _load_policies(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "shadow mode requires torch; sensor mode and diagnostics do not"
            ) from exc

        from .himloco_observation import HIMLocoObservation
        from .lidar_ray_adapter import LidarRayAdapter
        from .model_loader import infer, load_himloco_policy, load_navigation_policy
        from .navigation_observation import NavigationObservation

        self._torch = torch
        self._infer = infer
        self.device = torch.device(self.args.device)
        self.last_command = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.nav_policy = load_navigation_policy(self.args.navigation_policy, self.args.navigation_metadata, self.device)
        self.him_policy = load_himloco_policy(self.args.himloco_policy, self.device)
        print(f"[shadow] SEA-Nav policy={self.nav_policy.path} sha256={self.nav_policy.sha256}")
        print(f"[shadow] HIMLoco policy={self.him_policy.path} sha256={self.him_policy.sha256}")
        self.nav_obs = NavigationObservation(self.device)
        self.him_obs = HIMLocoObservation(self.device)
        self.ray_adapter = LidarRayAdapter(
            self.device, self.config.ray_count, self.config.ray_min_distance,
            self.config.ray_max_distance, self.config.ray_angle_min,
            self.config.ray_angle_max, self.config.lidar_min_z, self.config.lidar_max_z,
        )

    def run(self):
        print(f"[{self.args.mode}] read-only runtime started")
        print("[safety] LOWCMD DISABLED: no LowCmd import, publisher, mode switch, or motor write")
        try:
            while self.args.duration <= 0 or time.monotonic() - self.started < self.args.duration:
                cycle_start = time.perf_counter()
                self.reader.spin_once(0.0)
                if self.reader.goal.value is not None:
                    self.goal_manager.update(self.reader.goal.value)
                if self.args.mode == "sensor":
                    self._sensor_cycle(cycle_start)
                else:
                    self._shadow_cycle(cycle_start)
                elapsed = time.perf_counter() - cycle_start
                deadline_miss = elapsed > self.rate
                if not deadline_miss:
                    time.sleep(self.rate - elapsed)
        except KeyboardInterrupt:
            print(f"[{self.args.mode}] stopped")
        finally:
            self.command_bridge.assert_no_writes()
            self.reader.close()
            self.logger.close()

    def _sensor_cycle(self, cycle_start):
        health = self.reader.health(self.config.max_sensor_age)
        deadline_miss = _ms(cycle_start) > self.rate * 1000.0
        report = self.safety.evaluate(health["sensor_ages"], mode="sensor", values=health["finite_values"],
                                      deadline_miss=deadline_miss, wireless_emergency=health["wireless_emergency"])
        record = {"mode": "sensor", "timestamp": time.time(), "runtime_state": report.state.value,
                  "fault_reason": report.fault_reason, "sensor": _health_for_log(health),
                  "loop_latency_ms": _ms(cycle_start), "lowcmd_sent": False}
        if self._should_log() or report.state not in (RuntimeState.SENSOR, RuntimeState.SENSOR_WAIT):
            self.logger.write(record)

    def _shadow_cycle(self, cycle_start):
        health = self.reader.health(self.config.max_sensor_age)
        required = (self.reader.lowstate.value, self.reader.lidar.value, self.reader.odom.value,
                    self.goal_manager.current)
        if any(value is None for value in required):
            report = self.safety.evaluate(health["ages"], mode="shadow", values=health["finite_values"],
                                          wireless_emergency=health["wireless_emergency"])
            self._log_shadow_wait(health, report, cycle_start)
            return
        try:
            low = self.reader.lowstate.value
            odom = self.reader.odom.value
            goal = self.goal_manager.relative_xy(
                odom.position[:2], quat_to_yaw(low.quaternion), odom.frame_id, self.args.base_frame,
            )
            torch = self._torch
            gravity = torch.from_numpy(quat_to_gravity(low.quaternion)).reshape(1, 3).to(self.device)
            gyro = torch.from_numpy(low.gyro).reshape(1, 3).to(self.device)
            q_motor = torch.from_numpy(low.q_motor).reshape(1, 12).to(self.device)
            dq_motor = torch.from_numpy(low.dq_motor).reshape(1, 12).to(self.device)
            q_policy = q_motor[:, self.policy_to_motor]
            dq_policy = dq_motor[:, self.policy_to_motor]
            goal_xy = torch.from_numpy(goal).reshape(1, 2).to(self.device)
            linear_velocity = torch.from_numpy(odom.linear_velocity).reshape(1, 3).to(self.device)
            angular_velocity = gyro
            rays = self.ray_adapter.project(self.reader.lidar.value)

            nav_input = self.nav_obs.build(gravity, self.last_command, linear_velocity,
                                           angular_velocity, rays, goal_xy)
            nav_start = time.perf_counter()
            nav_raw = self._infer(self.nav_policy, nav_input, 3)
            nav_latency_ms = _ms(nav_start)
            nav_command = self.command_bridge.validate(nav_raw[0].detach().cpu().numpy())
            filtered_command = self.command_bridge.filter(nav_raw[0].detach().cpu().numpy())
            nav_command_t = torch.from_numpy(filtered_command).reshape(1, 3).to(self.device)
            him_input = self.him_obs.build(nav_command_t, angular_velocity, gravity,
                                           q_policy - torch.from_numpy(DEFAULT_JOINT_ANGLES).reshape(1, 12).to(self.device),
                                           dq_policy)
            him_start = time.perf_counter()
            him_action = self._infer(self.him_policy, him_input, 12)
            him_latency_ms = _ms(him_start)
            self.him_obs.record_action(him_action)
            self.last_command = nav_command_t.detach()
            deadline_miss = _ms(cycle_start) > self.rate * 1000.0
            report = self.safety.evaluate(health["ages"], values=[nav_input, him_input, nav_raw, him_action],
                                          mode="shadow", deadline_miss=deadline_miss,
                                          wireless_emergency=health["wireless_emergency"])
            record = {
                "mode": "shadow", "timestamp": time.time(), "runtime_state": report.state.value,
                "fault_reason": report.fault_reason, "goal": self.goal_manager.current,
                "goal_robot_xy": goal.tolist(), "odom": odom, "yaw": quat_to_yaw(low.quaternion),
                "lidar_valid": bool(np.isfinite(self.reader.lidar.value).all()),
                "sea_observation_shape": list(nav_input.shape), "sea_observation_min": float(nav_input.min()),
                "sea_observation_max": float(nav_input.max()), "sea_raw_command": nav_raw[0].detach().cpu().tolist(),
                "sea_clipped_command": nav_command.tolist(), "sea_filtered_command": filtered_command.tolist(),
                "command_filter_alpha": self.config.command_filter_alpha,
                "sea_inference_latency_ms": nav_latency_ms,
                "him_observation_shape": list(him_input.shape), "him_observation_min": float(him_input.min()),
                "him_observation_max": float(him_input.max()), "him_action": him_action[0].detach().cpu().tolist(),
                "him_inference_latency_ms": him_latency_ms, "sensor": _health_for_log(health),
                "loop_latency_ms": _ms(cycle_start), "deadline_miss": deadline_miss,
                "lowcmd_sent": False,
            }
            if self._should_log() or report.state not in (RuntimeState.SHADOW,):
                self.logger.write(record)
        except (FloatingPointError, ValueError, RuntimeError) as exc:
            report = self.safety.evaluate(health["ages"], values=(), mode="shadow",
                                          wireless_emergency=health["wireless_emergency"])
            self.logger.write({"mode": "shadow", "timestamp": time.time(), "runtime_state": RuntimeState.INVALID_DATA.value,
                               "fault_reason": str(exc), "sensor": _health_for_log(health), "lowcmd_sent": False})

    def _log_shadow_wait(self, health, report, cycle_start):
        if self._should_log():
            self.logger.write({"mode": "shadow", "timestamp": time.time(), "runtime_state": RuntimeState.SENSOR_WAIT.value,
                               "fault_reason": report.fault_reason or "waiting_for_lowstate_lidar_odom_goal",
                               "sensor": _health_for_log(health), "loop_latency_ms": _ms(cycle_start),
                               "lowcmd_sent": False})

    def _should_log(self):
        now = time.monotonic()
        if now - self.last_report >= self.args.log_interval:
            self.last_report = now
            return True
        return False


def _ms(start):
    return (time.perf_counter() - start) * 1000.0


def _health_for_log(health):
    """Keep JSONL bounded; raw numeric values remain available to safety checks."""
    return {key: value for key, value in health.items() if key != "finite_values"}


def build_parser():
    parser = argparse.ArgumentParser(description="Read-only Unitree Go2 sensor/shadow runtime")
    parser.add_argument("--mode", choices=("sensor", "shadow"), required=True)
    parser.add_argument("--navigation-policy")
    parser.add_argument("--navigation-metadata", default="")
    parser.add_argument("--himloco-policy")
    parser.add_argument("--lowstate-topic", default="/lowstate")
    parser.add_argument("--lidar-topic", default="/utlidar/cloud_base")
    parser.add_argument("--odom-topic", default="/utlidar/robot_odom")
    parser.add_argument("--wireless-topic", default="/wirelesscontroller")
    parser.add_argument("--goal-topic", default="/sea_nav/goal2d")
    parser.add_argument("--goal-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "cuda:0"))
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--max-sensor-age", type=float, default=0.25)
    parser.add_argument("--command-filter-alpha", type=float, default=1.0)
    parser.add_argument("--log-interval", type=float, default=1.0)
    parser.add_argument("--log", default="")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means until Ctrl-C")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "shadow" and (not args.navigation_policy or not args.himloco_policy):
        raise SystemExit("shadow mode requires --navigation-policy and --himloco-policy")
    if (args.goal_x is None) != (args.goal_y is None):
        raise SystemExit("provide both --goal-x and --goal-y, or neither")
    if args.device.startswith("cuda"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise SystemExit("CUDA/shadow mode requires torch") from exc
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
    OnboardRuntime(args).run()


if __name__ == "__main__":
    main()

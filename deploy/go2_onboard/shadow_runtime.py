"""Read-only SEA-Nav + HIMLoco Go2 onboard runtime.

This program subscribes to robot sensors and prints inference diagnostics. It
does not import Unitree LowCmd, create a lowcmd publisher, send torque/position
commands, or switch robot modes.
"""

import argparse
import time

import numpy as np
import torch

from .config import RuntimeConfig
from .himloco_observation import HIMLocoObservation
from .joint_mapping import make_policy_to_motor
from .lidar_ray_adapter import LidarRayAdapter
from .model_loader import infer, load_himloco_policy, load_navigation_policy
from .navigation_observation import NavigationObservation
from .ros_state_reader import RosStateReader
from .safety_monitor import SafetyMonitor
from .timing import FixedRate


class Topics:
    def __init__(self, args):
        self.lowstate = args.lowstate_topic
        self.lidar = args.lidar_topic
        self.odom = args.odom_topic
        self.wireless = args.wireless_topic


def _quat_to_gravity(quaternion):
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.size != 4:
        raise ValueError("IMU quaternion must contain 4 values in [w,x,y,z] order")
    w, x, y, z = [float(v) for v in q]
    # R^T * [0, 0, -1], matching quat_rotate_inverse(base_quat, gravity_vec).
    # Keep the same [w, x, y, z] convention as HIMLoco's deployment helper.
    return torch.tensor([[2.0 * (-z * x + w * y), -2.0 * (z * y + w * x),
                          1.0 - 2.0 * (w * w + z * z)]], dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="SEA-Nav Go2 read-only shadow runtime")
    parser.add_argument("--navigation-policy", required=True)
    parser.add_argument("--navigation-metadata", default="")
    parser.add_argument("--himloco-policy", required=True)
    parser.add_argument("--lowstate-topic", default="/lowstate")
    parser.add_argument("--lidar-topic", default="/utlidar/cloud_base")
    parser.add_argument("--odom-topic", default="/utlidar/robot_odom")
    parser.add_argument("--wireless-topic", default="/wirelesscontroller")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "cuda:0"))
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--goal-x", type=float, default=0.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--log-interval", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable; use --device cpu for local shadow testing")
    device = torch.device(args.device)
    config = RuntimeConfig(
        navigation_policy=args.navigation_policy,
        navigation_metadata=args.navigation_metadata,
        himloco_policy=args.himloco_policy,
        control_hz=args.control_hz,
        goal_xy=[args.goal_x, args.goal_y],
        log_interval=args.log_interval,
    )
    nav_policy = load_navigation_policy(config.navigation_policy, config.navigation_metadata, device)
    him_policy = load_himloco_policy(config.himloco_policy, device)
    nav_obs = NavigationObservation(device)
    him_obs = HIMLocoObservation(device)
    ray_adapter = LidarRayAdapter(device, config.ray_count, config.ray_min_distance,
                                  config.ray_max_distance, config.ray_angle_min,
                                  config.ray_angle_max, config.lidar_min_z, config.lidar_max_z)
    policy_to_motor = make_policy_to_motor()
    reader = RosStateReader(Topics(args))
    monitor = SafetyMonitor(config.max_sensor_age)
    rate = FixedRate(config.control_hz)
    last_command = torch.zeros((1, 3), dtype=torch.float32, device=device)
    goal_xy = torch.tensor([config.goal_xy], dtype=torch.float32, device=device)
    last_print = 0.0

    print("[shadow] read-only runtime started")
    print(f"[shadow] navigation policy: {nav_policy.path}")
    print(f"[shadow] HIMLoco policy: {him_policy.path}")
    print(f"[shadow] topics: lowstate={args.lowstate_topic} lidar={args.lidar_topic} odom={args.odom_topic} wireless={args.wireless_topic}")
    print(f"[shadow] device={device} control_hz={config.control_hz} goal_xy={config.goal_xy}")
    print("[shadow] LOWCMD DISABLED: no publisher, no mode switch, no motor command")

    try:
        while True:
            reader.spin_once(0.0)
            if reader.lowstate.value is None or reader.lidar.value is None or reader.odom.value is None:
                if time.monotonic() - last_print >= config.log_interval:
                    print(f"[shadow] waiting lowstate_age={reader.lowstate.age} lidar_age={reader.lidar.age} odom_age={reader.odom.age}")
                    last_print = time.monotonic()
                rate.sleep()
                continue

            low = reader.lowstate.value
            odom = reader.odom.value
            gravity = _quat_to_gravity(low.quaternion).to(device)
            gyro = torch.from_numpy(low.gyro).reshape(1, 3).to(device)
            q_motor = torch.from_numpy(low.q_motor).reshape(1, 12).to(device)
            dq_motor = torch.from_numpy(low.dq_motor).reshape(1, 12).to(device)
            q_policy = q_motor[:, policy_to_motor]
            dq_policy = dq_motor[:, policy_to_motor]
            joint_defaults = torch.tensor([[0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                                             0.1, 1.0, -1.5, -0.1, 1.0, -1.5]], device=device)
            rays = ray_adapter.project(reader.lidar.value)
            linear_velocity = torch.from_numpy(odom.linear_velocity).reshape(1, 3).to(device)
            angular_velocity = gyro
            nav_input = nav_obs.build(gravity, last_command, linear_velocity, angular_velocity, rays, goal_xy)
            t0 = time.perf_counter()
            nav_action = infer(nav_policy, nav_input, 3)
            nav_latency = (time.perf_counter() - t0) * 1000.0
            nav_command = nav_action.clamp(-config.max_nav_action, config.max_nav_action)
            nav_command = torch.maximum(nav_command, torch.tensor(config.command_bounds_min, device=device))
            nav_command = torch.minimum(nav_command, torch.tensor(config.command_bounds_max, device=device))
            him_input = him_obs.build(nav_command, angular_velocity, gravity,
                                      q_policy - joint_defaults, dq_policy)
            t1 = time.perf_counter()
            him_action = infer(him_policy, him_input, 12)
            him_latency = (time.perf_counter() - t1) * 1000.0
            him_obs.record_action(him_action)
            last_command = nav_command.detach()
            report = monitor.check({"lowstate": reader.lowstate.age, "lidar": reader.lidar.age, "odom": reader.odom.age}, nav_action, him_action)

            now = time.monotonic()
            if now - last_print >= config.log_interval or not report["finite"] or report["stale"]:
                print(f"[shadow] lowstate_age={report['ages']['lowstate']:.3f}s lidar_age={report['ages']['lidar']:.3f}s odom_age={report['ages']['odom']:.3f}s")
                print(f"[shadow] nav_obs={tuple(nav_input.shape)} nav_action={nav_action[0].detach().cpu().tolist()} nav_ms={nav_latency:.2f}")
                print(f"[shadow] him_obs={tuple(him_input.shape)} him_action_range=[{him_action.min().item():+.3f},{him_action.max().item():+.3f}] him_ms={him_latency:.2f} finite={report['finite']} stale={report['stale']}")
                print("[shadow] lowcmd_sent=False")
                last_print = now
            rate.sleep()
    finally:
        reader.close()


if __name__ == "__main__":
    main()

"""ROS-only Process A for the one-way laptop shadow runtime."""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from .goal import Goal2D, GoalManager
from .ipc_schema import encode_packet, make_packet
from .lidar_ray_adapter import NumpyLidarRayAdapter
from .ros_state_reader import RosStateReader
from .runtime import Topics, quat_to_gravity, quat_to_yaw


DEFAULT_SOCKET = "/tmp/sea_nav_shadow.sock"


class SensorBridge:
    def __init__(self, args):
        self.args = args
        self.reader = RosStateReader(Topics(args), max_sensor_age=args.max_sensor_age)
        self.goal_manager = GoalManager()
        if args.goal_x is not None and args.goal_y is not None:
            self.goal_manager.update(Goal2D(args.goal_frame, args.goal_x, args.goal_y, time.time()))
        self.rays = NumpyLidarRayAdapter()
        self.sequence = 0

    def snapshot(self):
        if self.reader.goal.value is not None:
            self.goal_manager.update(self.reader.goal.value)
        health = self.reader.health(self.args.max_sensor_age)
        low = self.reader.lowstate.value
        lidar = self.reader.lidar.value
        odom = self.reader.odom.value
        goal = self.goal_manager.current
        valid = {
            "lowstate": low is not None,
            "lidar": lidar is not None,
            "odom": odom is not None,
            "goal": goal is not None,
        }
        if low is None:
            low_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            joint_pos = np.zeros(12, dtype=np.float32)
            joint_vel = np.zeros(12, dtype=np.float32)
            gyro = np.zeros(3, dtype=np.float32)
        else:
            low_quaternion = low.quaternion
            joint_pos = low.q_motor
            joint_vel = low.dq_motor
            gyro = low.gyro
        gravity = quat_to_gravity(low_quaternion)
        rays = self.rays.project(lidar) if lidar is not None else np.full((1, 41), 5.0, dtype=np.float32)
        if odom is None:
            linear = np.zeros(3, dtype=np.float32)
            odom_angular = np.zeros(3, dtype=np.float32)
            position = np.zeros(2, dtype=np.float32)
            yaw = 0.0
            odom_frame = ""
        else:
            linear = odom.linear_velocity
            odom_angular = odom.angular_velocity
            position = odom.position[:2]
            yaw = quat_to_yaw(low_quaternion)
            odom_frame = odom.frame_id
        if goal is None:
            goal_body = np.zeros(2, dtype=np.float32)
        else:
            goal_body = self.goal_manager.relative_xy(position, yaw, odom_frame, self.args.base_frame)
        ages = health["ages"]
        packet = make_packet(
            sequence=self.sequence,
            timestamp_monotonic=time.monotonic(),
            timestamp_wall=time.time(),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            imu_ang_vel=gyro,
            projected_gravity=gravity,
            base_linear_velocity_body=linear,
            base_angular_velocity_body=gyro,
            lidar_rays=rays[0],
            goal_body=goal_body,
            sensor_age=ages,
            validity=valid,
        )
        self.sequence += 1
        return packet, health

    def close(self):
        self.reader.close()


def run(args):
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(1)
    server.settimeout(0.5)
    bridge = SensorBridge(args)
    started = time.monotonic()
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    connection = None
    print(f"[sensor_bridge] listening socket={args.socket}")
    print("[safety] ROS sensor reader only; no Torch, LowCmd, SportClient, or write path")
    try:
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            if connection is None:
                try:
                    connection, _ = server.accept()
                    connection.settimeout(1.0)
                    print("[sensor_bridge] shadow worker connected")
                except socket.timeout:
                    bridge.reader.spin_once(0.0)
                    continue
            cycle = time.perf_counter()
            bridge.reader.spin_once(0.0)
            packet, health = bridge.snapshot()
            try:
                connection.sendall(encode_packet(packet))
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                print("[sensor_bridge] worker disconnected; stopping")
                break
            record = {
                "mode": "sensor_bridge", "timestamp": time.time(), "sequence": packet["sequence"],
                "packet_validity": packet["validity"], "sensor_age": packet["sensor_age"],
                "packet_rate_hz": 1.0 / max(time.perf_counter() - cycle, 1e-6),
                "ipc_latency_ms": (time.perf_counter() - cycle) * 1000.0,
                "lowcmd_sent": False,
            }
            if log:
                log.write(json.dumps(record, separators=(",", ":")) + "\n")
                log.flush()
            print(json.dumps(record, separators=(",", ":")))
            elapsed = time.perf_counter() - cycle
            if elapsed < args.period:
                time.sleep(args.period - elapsed)
    except KeyboardInterrupt:
        print("[sensor_bridge] stopped")
    finally:
        if connection is not None:
            connection.close()
        server.close()
        bridge.close()
        if log:
            log.close()
        try:
            os.unlink(args.socket)
        except FileNotFoundError:
            pass


def build_parser():
    parser = argparse.ArgumentParser(description="ROS-only one-way SEA-Nav shadow sensor bridge")
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--max-sensor-age", type=float, default=0.25)
    parser.add_argument("--log", default="logs/go2_4d/sensor_bridge.jsonl")
    parser.add_argument("--lowstate-topic", default="/lowstate")
    parser.add_argument("--lidar-topic", default="/utlidar/cloud_base")
    parser.add_argument("--odom-topic", default="/utlidar/robot_odom")
    parser.add_argument("--wireless-topic", default="/wirelesscontroller")
    parser.add_argument("--goal-topic", default="/sea_nav/goal2d")
    parser.add_argument("--goal-frame", default="odom")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.set_defaults(period=1.0 / 50.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.goal_x is None) != (args.goal_y is None):
        raise SystemExit("provide both --goal-x and --goal-y, or neither")
    args.period = 1.0 / args.control_hz
    run(args)


if __name__ == "__main__":
    main()

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
from .ros_state_reader import RosStateReader
from .runtime import Topics, quat_to_gravity, quat_to_yaw
from .timing import FixedRate, NumericStats, RateStats


DEFAULT_SOCKET = "/tmp/sea_nav_shadow.sock"
DEFAULT_FRESHNESS = {
    "lowstate": 0.10,
    "odom": 0.10,
    "lidar": 0.20,
    "wireless": 0.25,
    "goal": 1.0,
}


def duration_expired(started, duration, now):
    """Apply bridge duration only after the shadow worker has connected."""
    return started is not None and duration > 0.0 and now - started >= duration


class SensorBridge:
    def __init__(self, args):
        self.args = args
        self.freshness = dict(DEFAULT_FRESHNESS)
        if args.max_sensor_age is not None:
            self.freshness = {name: float(args.max_sensor_age) for name in self.freshness}
        for name in ("lowstate", "odom", "lidar", "wireless", "goal"):
            value = getattr(args, f"{name}_max_age")
            if value is not None:
                self.freshness[name] = float(value)
        self.reader = RosStateReader(Topics(args), max_sensor_age=max(self.freshness.values()))
        self.goal_manager = GoalManager()
        if args.goal_x is not None and args.goal_y is not None:
            self.goal_manager.update(Goal2D(args.goal_frame, args.goal_x, args.goal_y, time.time()))
        self.sequence = 0

    def snapshot(self):
        snapshot_start = time.perf_counter()
        with self.reader.lock:
            if self.reader.goal.value is not None:
                self.goal_manager.update(self.reader.goal.value)
            health = self.reader.health(sensor_max_ages=self.freshness)
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
        lidar_cache_start = time.perf_counter()
        lidar_cache = self.reader.lidar_cache.snapshot()
        rays = lidar_cache["rays"] if lidar is not None else np.full((41,), 5.0, dtype=np.float32)
        lidar_cache_copy_ms = (time.perf_counter() - lidar_cache_start) * 1000.0
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
            lidar_rays=rays,
            goal_body=goal_body,
            sensor_age=ages,
            validity=valid,
        )
        self.sequence += 1
        return packet, health, {
            "snapshot_build_ms": (time.perf_counter() - snapshot_start) * 1000.0,
            "lidar_processing_ms": 0.0,
            "lidar_cache_copy_ms": lidar_cache_copy_ms,
            "lidar_callback_processing_ms": lidar_cache["processing_ms"],
            "lidar_cache_processed_count": lidar_cache["processed_count"],
            "lidar_cache_source_timestamp": lidar_cache["source_timestamp"],
        }

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
    bridge.reader.start_background_spin()
    started = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    connection = None
    rate_stats = RateStats()
    snapshot_stats = NumericStats()
    lidar_stats = NumericStats()
    encode_stats = NumericStats()
    send_stats = NumericStats()
    log_stats = NumericStats()
    scheduler = None
    scheduler_deadline_misses = 0
    previous_sleep_ms = 0.0
    last_log_write_ms = 0.0
    last_summary = time.monotonic()
    print(f"[sensor_bridge] listening socket={args.socket}")
    print("[safety] ROS sensor reader only; no Torch, LowCmd, SportClient, or write path")
    try:
        while not duration_expired(started, args.duration, time.monotonic()):
            if connection is None:
                try:
                    connection, _ = server.accept()
                    connection.settimeout(1.0)
                    started = time.monotonic()
                    scheduler = FixedRate(args.control_hz)
                    print("[sensor_bridge] shadow worker connected")
                except socket.timeout:
                    continue
            cycle = time.perf_counter()
            packet, health, timing = bridge.snapshot()
            snapshot_stats.add(timing["snapshot_build_ms"])
            lidar_stats.add(timing["lidar_processing_ms"])
            encode_start = time.perf_counter()
            payload = encode_packet(packet)
            encode_ms = (time.perf_counter() - encode_start) * 1000.0
            encode_stats.add(encode_ms)
            send_start = time.perf_counter()
            try:
                connection.sendall(payload)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                print("[sensor_bridge] worker disconnected; stopping")
                break
            send_ms = (time.perf_counter() - send_start) * 1000.0
            send_stats.add(send_ms)
            send_finished = time.perf_counter()
            rate_stats.observe(send_finished)
            record = {
                "mode": "sensor_bridge", "timestamp": time.time(), "sequence": packet["sequence"],
                "packet_validity": packet["validity"], "sensor_age": packet["sensor_age"],
                "freshness_thresholds": health["sensor_thresholds"],
                "packet_rate_hz": rate_stats.current_rate_hz(),
                "snapshot_build_ms": timing["snapshot_build_ms"],
                "lidar_processing_ms": timing["lidar_processing_ms"],
                "lidar_cache_copy_ms": timing["lidar_cache_copy_ms"],
                "lidar_callback_processing_ms": timing["lidar_callback_processing_ms"],
                "lidar_cache_processed_count": timing["lidar_cache_processed_count"],
                "lidar_cache_source_timestamp": timing["lidar_cache_source_timestamp"],
                "ipc_encode_ms": encode_ms,
                "ipc_send_ms": send_ms,
                "log_write_ms": last_log_write_ms,
                "loop_total_ms": (send_finished - cycle) * 1000.0,
                "sleep_ms": previous_sleep_ms,
                "deadline_miss": (send_finished - cycle) > args.period,
                "ipc_latency_ms": (send_finished - cycle) * 1000.0,
                "lowcmd_sent": False,
            }
            log_start = time.perf_counter()
            if log:
                log.write(json.dumps(record, separators=(",", ":")) + "\n")
            last_log_write_ms = (time.perf_counter() - log_start) * 1000.0
            log_stats.add(last_log_write_ms)
            schedule = scheduler.sleep()
            previous_sleep_ms = schedule["sleep_s"] * 1000.0
            if schedule["deadline_miss"]:
                scheduler_deadline_misses += 1
            if time.monotonic() - last_summary >= args.summary_interval:
                summary = {"mode": "sensor_bridge", "state": "RUNNING", **rate_stats.summary()}
                summary.update({
                    "snapshot_p95_ms": snapshot_stats.percentile(0.95),
                    "lidar_p95_ms": lidar_stats.percentile(0.95),
                    "encode_p95_ms": encode_stats.percentile(0.95),
                    "send_p95_ms": send_stats.percentile(0.95),
                    "log_p95_ms": log_stats.percentile(0.95),
                    "deadline_misses": scheduler_deadline_misses,
                    "lowcmd_sent": False,
                })
                print("[sensor_bridge] " + json.dumps(summary, separators=(",", ":")))
                last_summary = time.monotonic()
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
    parser.add_argument("--summary-interval", type=float, default=1.0)
    parser.add_argument("--max-sensor-age", type=float, default=None,
                        help="legacy override; prefer per-sensor freshness options")
    parser.add_argument("--lowstate-max-age", type=float, default=None)
    parser.add_argument("--odom-max-age", type=float, default=None)
    parser.add_argument("--lidar-max-age", type=float, default=None)
    parser.add_argument("--wireless-max-age", type=float, default=None)
    parser.add_argument("--goal-max-age", type=float, default=None)
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

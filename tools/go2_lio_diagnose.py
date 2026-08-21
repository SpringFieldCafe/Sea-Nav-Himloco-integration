#!/usr/bin/env python3
"""Read-only ROS2 monitor for the independent Go2 Point-LIO pipeline."""

import argparse
import math
import os
import subprocess
import time
from dataclasses import dataclass


@dataclass
class Stream:
    count: int = 0
    first: float = 0.0
    last: float = 0.0
    last_interval: float = 0.0
    frame: str = ""
    child: str = ""
    pose: tuple | None = None

    def note(self, now, frame="", child="", pose=None):
        if self.last:
            self.last_interval = now - self.last
        if not self.first:
            self.first = now
        self.last = now
        self.count += 1
        if frame:
            self.frame = frame
        if child:
            self.child = child
        if pose is not None:
            self.pose = pose

    def age(self, now):
        return None if not self.last else max(0.0, now - self.last)

    def rate(self):
        if self.count < 2 or self.last <= self.first:
            return 0.0
        return (self.count - 1) / (self.last - self.first)


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def angle_delta(a, b):
    return (b - a + math.pi) % (2.0 * math.pi) - math.pi


def command_text(command, timeout=2.0):
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout


def process_rows():
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            text=True,
            capture_output=True,
            timeout=1.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            rows.append((int(parts[0]), parts[1]))
    return rows


def matching_pids(patterns):
    rows = process_rows()
    return [
        pid for pid, args in rows
        if any(pattern in args for pattern in patterns)
        and "go2_lio_diagnose.py" not in args
        and "go2_lio_selfcheck.sh" not in args
    ]


def pidfile_alive(path):
    try:
        pid = int(open(path, encoding="utf-8").read().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def publisher_count(node, topic):
    try:
        return int(node.count_publishers(topic))
    except Exception:
        return 0


def bridge_subscribed():
    output = command_text(
        ["ros2", "node", "info", "/sea_nav_go2_shadow_runtime"],
        timeout=2.0,
    )
    return "/sea_nav/lio/odom_base" in output


def process_state(pidfile, patterns):
    return pidfile_alive(pidfile) or bool(matching_pids(patterns))


class Monitor:
    def __init__(self, args):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, PointCloud2
        from unitree_go.msg import LowState

        self.args = args
        self.rclpy = rclpy
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        self.node = rclpy.create_node("go2_lio_selfcheck_diagnostic")
        self.streams = {
            "lowstate": Stream(),
            "cloud": Stream(),
            "imu": Stream(),
            "lio": Stream(),
            "base": Stream(),
        }
        self.node.create_subscription(
            LowState, "/lowstate", lambda msg: self.note("lowstate"), 10
        )
        self.node.create_subscription(
            PointCloud2, "/utlidar/cloud", lambda msg: self.note("cloud"), 10
        )
        self.node.create_subscription(
            Imu, "/utlidar/imu", lambda msg: self.note("imu"), 10
        )
        self.node.create_subscription(
            Odometry,
            "/sea_nav/lio/odom",
            lambda msg: self.note_odom("lio", msg),
            10,
        )
        self.node.create_subscription(
            Odometry,
            "/sea_nav/lio/odom_base",
            lambda msg: self.note_odom("base", msg),
            10,
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.started = time.monotonic()
        self.drift_started = None
        self.drift_start_pose = None
        self.drift_done = False
        self.drift_result = "WAITING"
        self.drift_failure_reason = ""

    def note(self, name):
        self.streams[name].note(time.monotonic())

    def note_odom(self, name, msg):
        pose = msg.pose.pose
        self.streams[name].note(
            time.monotonic(),
            str(msg.header.frame_id or ""),
            str(msg.child_frame_id or ""),
            (float(pose.position.x), float(pose.position.y), yaw_from_quaternion(pose.orientation)),
        )

    def ready_for_drift(self, now):
        return all(
            self.streams[name].count > 0
            and (self.streams[name].age(now) or 999.0) < 1.0
            for name in ("lowstate", "cloud", "imu", "lio", "base")
        )

    def component_info(self, now):
        lio_pubs = publisher_count(self.node, "/sea_nav/lio/odom")
        base_pubs = publisher_count(self.node, "/sea_nav/lio/odom_base")
        nodes = command_text(["ros2", "node", "list"], timeout=2.0)
        return {
            "point_running": process_state(
                self.args.point_pid,
                ("pointlio_mapping", "transform_everything", "point_lio_go2.launch.py"),
            ) or "/sea_nav_point_lio" in nodes,
            "adapter_running": process_state(
                self.args.adapter_pid, ("odom_se2_adapter",)
            ) or "/sea_nav_lio_odom_se2_adapter" in nodes,
            "bridge_running": process_state(
                self.args.bridge_pid, ("deploy.go2_onboard.sensor_bridge",)
            ) or "/sea_nav_go2_shadow_runtime" in nodes,
            "lio_pubs": lio_pubs,
            "base_pubs": base_pubs,
            "bridge_sub": bridge_subscribed(),
        }

    def diagnosis(self, info, now):
        if self.drift_failure_reason:
            return self.drift_failure_reason
        if any(self.streams[name].count == 0 for name in ("lowstate", "cloud", "imu")):
            return "RAW_SENSOR_FAILURE"
        if info["lio_pubs"] > 1 or info["base_pubs"] > 1:
            return "DUPLICATE_PUBLISHER"
        if not info["point_running"]:
            return "POINT_LIO_EXITED"
        if self.streams["lio"].count == 0:
            return "POINT_LIO_NO_OUTPUT"
        if not info["adapter_running"]:
            return "ADAPTER_NOT_RUNNING"
        if self.streams["base"].count == 0:
            return "ADAPTER_NO_OUTPUT"
        if self.streams["base"].frame != "odom" or self.streams["base"].child != "base_link":
            return "ADAPTER_BAD_FRAME"
        if not info["bridge_running"]:
            return "SENSOR_BRIDGE_NOT_RUNNING"
        if not info["bridge_sub"]:
            return "SENSOR_BRIDGE_ODOM_NOT_SUBSCRIBED"
        if any(self.streams[name].count == 0 for name in ("lio", "base")):
            return "DDS_ENVIRONMENT_MISMATCH"
        return "PASS"

    def drift(self, now):
        if self.drift_done:
            return None
        if self.drift_started is None:
            if not self.ready_for_drift(now):
                return None
            self.drift_started = now
            self.drift_start_pose = (
                self.streams["lio"].pose,
                self.streams["base"].pose,
            )
            print("DRIFT_CHECK_START=10s", flush=True)
            return None
        if now - self.drift_started < 10.0:
            return None
        raw0, base0 = self.drift_start_pose
        raw1, base1 = self.streams["lio"].pose, self.streams["base"].pose
        if not raw0 or not base0 or not raw1 or not base1:
            self.drift_result = "FAIL"
            self.drift_failure_reason = "POINT_LIO_DRIFT"
            return "DRIFT_CHECK=FAIL"
        raw_pos = math.hypot(raw1[0] - raw0[0], raw1[1] - raw0[1])
        base_pos = math.hypot(base1[0] - base0[0], base1[1] - base0[1])
        raw_yaw = math.degrees(angle_delta(raw0[2], raw1[2]))
        base_yaw = math.degrees(angle_delta(base0[2], base1[2]))
        print(f"RAW_POSITION_DRIFT_CM={raw_pos * 100.0:.3f}", flush=True)
        print(f"RAW_YAW_DRIFT_DEG={raw_yaw:.3f}", flush=True)
        print(f"BASE_POSITION_DRIFT_CM={base_pos * 100.0:.3f}", flush=True)
        print(f"BASE_YAW_DRIFT_DEG={base_yaw:.3f}", flush=True)
        if raw_pos > 0.05 or abs(raw_yaw) > 2.0:
            self.drift_result = "FAIL"
            self.drift_failure_reason = "POINT_LIO_DRIFT"
            print("DRIFT_CHECK=FAIL", flush=True)
        elif base_pos > 0.05 or abs(base_yaw) > 2.0:
            self.drift_result = "FAIL"
            self.drift_failure_reason = "ADAPTER_PROBLEM"
            print("DRIFT_CHECK=FAIL", flush=True)
        else:
            self.drift_result = "PASS"
            print("DRIFT_CHECK=PASS", flush=True)
        yaw = base1[2]
        print(f"CURRENT_BASE_X={base1[0]:.6f}", flush=True)
        print(f"CURRENT_BASE_Y={base1[1]:.6f}", flush=True)
        print(f"CURRENT_BASE_YAW_DEG={math.degrees(yaw):.3f}", flush=True)
        print(f"GOAL_0P5M_X={base1[0] + 0.5 * math.cos(yaw):.6f}", flush=True)
        print(f"GOAL_0P5M_Y={base1[1] + 0.5 * math.sin(yaw):.6f}", flush=True)
        self.drift_done = True
        return None

    def render(self, info, diagnosis, now):
        def state(name):
            stream = self.streams[name]
            return "PASS" if stream.count else "FAIL"

        lines = [
            "================ GO2 LIO STATUS ================",
            f"RAW_LOWSTATE      = {state('lowstate')}",
            f"RAW_CLOUD         = {state('cloud')}",
            f"RAW_IMU           = {state('imu')}",
            f"RAW_LOWSTATE_COUNT= {self.streams['lowstate'].count}",
            f"RAW_CLOUD_COUNT   = {self.streams['cloud'].count}",
            f"RAW_IMU_COUNT     = {self.streams['imu'].count}",
            f"POINT_LIO_PROCESS = {'RUNNING' if info['point_running'] else 'EXITED'}",
            f"POINT_LIO_PUBS    = {info['lio_pubs']}",
            f"POINT_LIO_DATA    = {state('lio')}",
            f"POINT_LIO_RATE    = {self.streams['lio'].rate():.1f} Hz",
            f"POINT_LIO_COUNT   = {self.streams['lio'].count}",
            f"ADAPTER_PROCESS   = {'RUNNING' if info['adapter_running'] else 'EXITED'}",
            f"ODOM_BASE_PUBS    = {info['base_pubs']}",
            f"ODOM_BASE_DATA    = {state('base')}",
            f"ODOM_BASE_RATE    = {self.streams['base'].rate():.1f} Hz",
            f"ODOM_BASE_COUNT   = {self.streams['base'].count}",
            f"ODOM_BASE_FRAME   = {self.streams['base'].frame or '<none>'} -> {self.streams['base'].child or '<none>'}",
            f"SENSOR_BRIDGE     = {'RUNNING' if info['bridge_running'] else 'EXITED'}",
            f"BRIDGE_ODOM_SUB   = {'PASS' if info['bridge_sub'] else 'FAIL'}",
            f"DIAGNOSIS         = {diagnosis}",
            f"DIAGNOSIS={diagnosis}",
            f"DRIFT_CHECK       = {self.drift_result}",
            "===================================================",
        ]
        print("\n".join(lines), flush=True)
        with open(self.args.status_file, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")

    def run(self):
        deadline = self.started + self.args.duration if self.args.duration > 0 else None
        next_report = 0.0
        try:
            while self.rclpy.ok() and (deadline is None or time.monotonic() < deadline):
                self.executor.spin_once(timeout_sec=0.2)
                now = time.monotonic()
                if now >= next_report:
                    info = self.component_info(now)
                    self.drift(now)
                    diagnosis = self.diagnosis(info, now)
                    self.render(info, diagnosis, now)
                    next_report = now + max(0.5, self.args.report_interval)
        finally:
            try:
                self.executor.shutdown()
            except Exception:
                pass
            try:
                self.node.destroy_node()
            except Exception:
                pass
            if self._owns_rclpy and self.rclpy.ok():
                self.rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--report-interval", type=float, default=2.0)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--point-pid", required=True)
    parser.add_argument("--adapter-pid", required=True)
    parser.add_argument("--bridge-pid", required=True)
    args = parser.parse_args()
    try:
        Monitor(args).run()
    except KeyboardInterrupt:
        print("[DIAGNOSTIC] Ctrl+C received; shutting down read-only monitor", flush=True)


if __name__ == "__main__":
    main()

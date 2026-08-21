#!/usr/bin/env python3
"""Read-only Point-LIO pose jump recorder for staged Go2 diagnostics."""

import argparse
from collections import deque
import json
import math
import select
import signal
import sys
import threading
import time


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def angle_delta(current, previous):
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


class PoseStream:
    def __init__(self, name):
        self.name = name
        self.last = None
        self.history = deque()
        self.max_single_position = 0.0
        self.max_single_yaw = 0.0
        self.max_window_position = {0.1: 0.0, 1.0: 0.0}
        self.max_window_yaw = {0.1: 0.0, 1.0: 0.0}
        self.samples = 0
        self.first_stamp = None
        self.last_stamp = None

    def update(self, stamp, x, y, z, yaw, receive_time):
        sample = (float(stamp), float(x), float(y), float(z), float(yaw))
        if self.first_stamp is None:
            self.first_stamp = sample[0]
        self.last_stamp = sample[0]
        if self.last is not None:
            _, px, py, pz, pyaw = self.last
            self.max_single_position = max(
                self.max_single_position,
                math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2),
            )
            self.max_single_yaw = max(self.max_single_yaw, abs(angle_delta(yaw, pyaw)))
        self.last = sample
        self.history.append((receive_time, sample))
        while self.history and receive_time - self.history[0][0] > 1.0:
            self.history.popleft()
        for window in self.max_window_position:
            old = self.history[0]
            for candidate in self.history:
                if receive_time - candidate[0] >= window:
                    old = candidate
                else:
                    break
            _, ox, oy, oz, oyaw = old[1]
            self.max_window_position[window] = max(
                self.max_window_position[window],
                math.sqrt((x - ox) ** 2 + (y - oy) ** 2 + (z - oz) ** 2),
            )
            self.max_window_yaw[window] = max(
                self.max_window_yaw[window], abs(angle_delta(yaw, oyaw))
            )
        self.samples += 1

    def summary(self):
        return {
            "samples": self.samples,
            "first_stamp": self.first_stamp,
            "last_stamp": self.last_stamp,
            "max_single_frame_position_jump_m": self.max_single_position,
            "max_single_frame_yaw_jump_deg": math.degrees(self.max_single_yaw),
            "max_100ms_position_jump_m": self.max_window_position[0.1],
            "max_100ms_yaw_jump_deg": math.degrees(self.max_window_yaw[0.1]),
            "max_1s_position_jump_m": self.max_window_position[1.0],
            "max_1s_yaw_jump_deg": math.degrees(self.max_window_yaw[1.0]),
        }


class Recorder:
    def __init__(self, args):
        import rclpy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu, PointCloud2

        self.rclpy = rclpy
        self.args = args
        self.node = rclpy.create_node("go2_lio_pose_jump_recorder")
        self.streams = {"lio": PoseStream("lio"), "base": PoseStream("base")}
        self.imu_count = 0
        self.cloud_count = 0
        self.imu_first_stamp = None
        self.imu_last_stamp = None
        self.cloud_first_stamp = None
        self.cloud_last_stamp = None
        self.started = time.monotonic()
        self.stop_event = threading.Event()
        self.output = open(args.output, "a", encoding="utf-8") if args.output else None
        self.node.create_subscription(Odometry, args.lio_topic, self._lio, 10)
        self.node.create_subscription(Odometry, args.base_topic, self._base, 10)
        self.node.create_subscription(Imu, args.imu_topic, self._imu, 50)
        if args.cloud_topic:
            self.node.create_subscription(PointCloud2, args.cloud_topic, self._cloud, 10)

    @staticmethod
    def _stamp(msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        return stamp if stamp > 0.0 else time.monotonic()

    def _odom(self, name, msg):
        pose = msg.pose.pose
        stamp = self._stamp(msg)
        receive_time = time.monotonic()
        self.streams[name].update(
            stamp,
            pose.position.x,
            pose.position.y,
            pose.position.z,
            yaw_from_quaternion(pose.orientation),
            receive_time,
        )
        self.write({
            "type": "pose",
            "stream": name,
            "timestamp": stamp,
            "receive_monotonic": receive_time,
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
            "yaw_rad": self.streams[name].last[4],
        })

    def _lio(self, msg):
        self._odom("lio", msg)

    def _base(self, msg):
        self._odom("base", msg)

    def _imu(self, msg):
        stamp = self._stamp(msg)
        self.imu_count += 1
        self.imu_first_stamp = stamp if self.imu_first_stamp is None else self.imu_first_stamp
        self.imu_last_stamp = stamp
        self.write({"type": "imu", "timestamp": stamp, "receive_monotonic": time.monotonic()})

    def _cloud(self, msg):
        stamp = self._stamp(msg)
        self.cloud_count += 1
        self.cloud_first_stamp = stamp if self.cloud_first_stamp is None else self.cloud_first_stamp
        self.cloud_last_stamp = stamp
        self.write({"type": "cloud", "timestamp": stamp, "receive_monotonic": time.monotonic()})

    def write(self, record):
        if self.output:
            self.output.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.output.flush()

    def marker_loop(self):
        while not self.stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not ready:
                continue
            marker = sys.stdin.readline().strip().upper()
            if marker in {"P", "S", "A", "Q"}:
                record = {"type": "marker", "marker": marker, "timestamp": time.time()}
                print("MARKER=" + marker, flush=True)
                self.write(record)
                if marker == "Q":
                    self.stop_event.set()
                    self.rclpy.shutdown()
                    return

    def run(self):
        marker_thread = threading.Thread(target=self.marker_loop, daemon=True)
        marker_thread.start()
        try:
            while self.rclpy.ok() and not self.stop_event.is_set():
                self.rclpy.spin_once(self.node, timeout_sec=0.2)
        finally:
            self.stop_event.set()
            if self.output:
                self.output.close()
            print("SUMMARY=" + json.dumps({name: stream.summary() for name, stream in self.streams.items()}))
            print(
                "SENSOR_STAMPS=" + json.dumps({
                    "imu_count": self.imu_count,
                    "imu_first": self.imu_first_stamp,
                    "imu_last": self.imu_last_stamp,
                    "cloud_count": self.cloud_count,
                    "cloud_first": self.cloud_first_stamp,
                    "cloud_last": self.cloud_last_stamp,
                }, separators=(",", ":"))
            )
            self.node.destroy_node()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lio-topic", default="/sea_nav/lio/odom")
    parser.add_argument("--base-topic", default="/sea_nav/lio/odom_base")
    parser.add_argument("--imu-topic", default="/utlidar/imu")
    parser.add_argument("--cloud-topic", default="/utlidar/cloud")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    import rclpy

    rclpy.init(args=None)
    recorder = Recorder(args)

    def stop(_signum, _frame):
        recorder.stop_event.set()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    recorder.run()


if __name__ == "__main__":
    main()

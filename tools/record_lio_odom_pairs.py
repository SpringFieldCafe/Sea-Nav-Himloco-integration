#!/usr/bin/env python3
"""Read-only recorder for time-matched native and independent odometry."""

import argparse
import json
import math
import threading
import time
from pathlib import Path


def stamp_seconds(stamp):
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def pose_from_odom(msg):
    pose = msg.pose.pose
    position = pose.position
    orientation = pose.orientation
    return {
        "x": float(position.x),
        "y": float(position.y),
        "z": float(position.z),
        "yaw": yaw_from_quaternion(orientation),
        "frame_id": str(msg.header.frame_id or ""),
        "child_frame_id": str(msg.child_frame_id or ""),
    }


def match_by_timestamp(native, lio, max_delta_s):
    """Greedily match each native sample to one nearest unused LIO sample."""
    native = sorted(native, key=lambda sample: sample["stamp"])
    lio = sorted(lio, key=lambda sample: sample["stamp"])
    matches = []
    used = set()
    for native_sample in native:
        if not lio:
            break
        candidates = (
            (abs(native_sample["stamp"] - sample["stamp"]), index, sample)
            for index, sample in enumerate(lio)
            if index not in used
        )
        try:
            delta, index, lio_sample = min(candidates, key=lambda item: item[0])
        except ValueError:
            break
        if delta <= max_delta_s:
            used.add(index)
            matches.append({
                "stamp": native_sample["stamp"],
                "native_stamp": native_sample["stamp"],
                "lio_stamp": lio_sample["stamp"],
                "sync_delta_s": delta,
                "native": native_sample["pose"],
                "lio": lio_sample["pose"],
            })
    return matches


class OdomRecorder:
    def __init__(self, native_topic, lio_topic):
        import rclpy
        from nav_msgs.msg import Odometry

        self._lock = threading.Lock()
        self.native = []
        self.lio = []
        self.node = rclpy.create_node("sea_nav_lio_odom_recorder")
        self.node.create_subscription(
            Odometry, native_topic, self._native_callback, 20
        )
        self.node.create_subscription(
            Odometry, lio_topic, self._lio_callback, 20
        )

    def _native_callback(self, msg):
        sample = {"stamp": stamp_seconds(msg.header.stamp), "pose": pose_from_odom(msg)}
        with self._lock:
            self.native.append(sample)

    def _lio_callback(self, msg):
        sample = {"stamp": stamp_seconds(msg.header.stamp), "pose": pose_from_odom(msg)}
        with self._lock:
            self.lio.append(sample)

    def snapshot(self):
        with self._lock:
            return list(self.native), list(self.lio)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-topic", default="/utlidar/robot_odom")
    parser.add_argument("--lio-topic", default="/sea_nav/lio/odom")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--max-sync-delta-s", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.duration <= 0.0 or args.max_sync_delta_s <= 0.0:
        raise SystemExit("duration and max-sync-delta-s must be positive")

    import rclpy

    rclpy.init(args=None)
    recorder = OdomRecorder(args.native_topic, args.lio_topic)
    deadline = time.monotonic() + args.duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(recorder.node, timeout_sec=0.05)
    finally:
        native, lio = recorder.snapshot()
        matches = match_by_timestamp(native, lio, args.max_sync_delta_s)
        payload = {
            "native_topic": args.native_topic,
            "lio_topic": args.lio_topic,
            "duration_s": args.duration,
            "max_sync_delta_s": args.max_sync_delta_s,
            "native_sample_count": len(native),
            "lio_sample_count": len(lio),
            "matched_sample_count": len(matches),
            "records": matches,
        }
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        recorder.node.destroy_node()
        rclpy.shutdown()
        print(f"NATIVE_SAMPLES={len(native)}")
        print(f"LIO_SAMPLES={len(lio)}")
        print(f"MATCHED_SAMPLES={len(matches)}")
        print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()

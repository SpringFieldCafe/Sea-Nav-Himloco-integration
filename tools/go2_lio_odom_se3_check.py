#!/usr/bin/env python3
"""Read-only static comparison of raw and SE(3)-adapted odometry."""

import argparse
import math
import time


def _normalize_quaternion(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12 or not math.isfinite(norm):
        raise ValueError("invalid quaternion")
    return x / norm, y / norm, z / norm, w / norm


def quaternion_rpy(q):
    x, y, z, w = _normalize_quaternion(*q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, pitch_arg)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def shortest_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def drift(samples):
    if len(samples) < 2:
        raise ValueError("not enough odometry samples")
    first = samples[0]
    last = samples[-1]
    dx = last[1][0] - first[1][0]
    dy = last[1][1] - first[1][1]
    dz = last[1][2] - first[1][2]
    position_drift = math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw_drift = abs(shortest_angle(last[2][2] - first[2][2]))
    rate = (len(samples) - 1) / (samples[-1][0] - samples[0][0])
    return position_drift, math.degrees(yaw_drift), rate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--raw-topic", default="/sea_nav/lio/odom")
    parser.add_argument("--base-topic", default="/sea_nav/lio/odom_base")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

    samples = {"raw": [], "base": []}
    base_rpy = []
    errors = []

    rclpy.init(args=None)
    node = rclpy.create_node("go2_lio_odom_se3_check")
    qos = QoSProfile(depth=50)
    qos.reliability = ReliabilityPolicy.RELIABLE
    qos.durability = DurabilityPolicy.VOLATILE

    def callback(kind, msg):
        try:
            position = (
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                float(msg.pose.pose.position.z),
            )
            rpy = quaternion_rpy((
                float(msg.pose.pose.orientation.x),
                float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z),
                float(msg.pose.pose.orientation.w),
            ))
            receive_time = time.monotonic()
            samples[kind].append((receive_time, position, rpy))
            if kind == "base":
                base_rpy.append(rpy)
        except (TypeError, ValueError, ArithmeticError) as exc:
            errors.append(str(exc))

    node.create_subscription(Odometry, args.raw_topic, lambda msg: callback("raw", msg), qos)
    node.create_subscription(Odometry, args.base_topic, lambda msg: callback("base", msg), qos)

    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if errors:
        raise SystemExit("invalid odometry sample: " + errors[-1])
    if len(samples["raw"]) < 2 or len(samples["base"]) < 2:
        raise SystemExit(
            "need at least two samples on both topics: raw=%d base=%d"
            % (len(samples["raw"]), len(samples["base"]))
        )

    raw_position_drift, raw_yaw_drift, raw_rate = drift(samples["raw"])
    base_position_drift, base_yaw_drift, base_rate = drift(samples["base"])
    base_roll_start, base_pitch_start, _ = samples["base"][0][2]
    base_roll_end, base_pitch_end, _ = samples["base"][-1][2]

    raw_delta = tuple(
        samples["raw"][-1][1][i] - samples["raw"][0][1][i] for i in range(3)
    )
    base_delta = tuple(
        samples["base"][-1][1][i] - samples["base"][0][1][i] for i in range(3)
    )
    extra_delta = tuple(base_delta[i] - raw_delta[i] for i in range(3))
    extra_position_drift = math.sqrt(sum(value * value for value in extra_delta))
    extra_yaw_drift = abs(
        math.degrees(shortest_angle(
            (samples["base"][-1][2][2] - samples["base"][0][2][2])
            - (samples["raw"][-1][2][2] - samples["raw"][0][2][2])
        ))
    )

    rate_match = min(raw_rate, base_rate) / max(raw_rate, base_rate) >= 0.85
    finite = all(math.isfinite(value) for value in (
        raw_position_drift, base_position_drift, raw_yaw_drift, base_yaw_drift,
        raw_rate, base_rate, extra_position_drift, extra_yaw_drift,
        base_roll_start, base_pitch_start, base_roll_end, base_pitch_end,
    ))
    base_attitude_continuous = all(
        abs(math.degrees(shortest_angle(base_rpy[i][j] - base_rpy[i - 1][j]))) < 20.0
        for i in range(1, len(base_rpy))
        for j in range(2)
    )
    contract_pass = (
        finite
        and rate_match
        and extra_position_drift <= 0.10
        and extra_yaw_drift <= 5.0
        and base_attitude_continuous
    )

    print("RAW_POSITION_DRIFT_M=%.6f" % raw_position_drift)
    print("BASE_POSITION_DRIFT_M=%.6f" % base_position_drift)
    print("RAW_YAW_DRIFT_DEG=%.6f" % raw_yaw_drift)
    print("BASE_YAW_DRIFT_DEG=%.6f" % base_yaw_drift)
    print("RAW_ODOM_RATE_HZ=%.6f" % raw_rate)
    print("BASE_ODOM_RATE_HZ=%.6f" % base_rate)
    print("BASE_ROLL_START_DEG=%.6f" % math.degrees(base_roll_start))
    print("BASE_PITCH_START_DEG=%.6f" % math.degrees(base_pitch_start))
    print("BASE_ROLL_END_DEG=%.6f" % math.degrees(base_roll_end))
    print("BASE_PITCH_END_DEG=%.6f" % math.degrees(base_pitch_end))
    print("ADAPTER_EXTRA_POSITION_DRIFT_M=%.6f" % extra_position_drift)
    print("ADAPTER_EXTRA_YAW_DRIFT_DEG=%.6f" % extra_yaw_drift)
    print("SE3_ADAPTER_STATIC_CONTRACT=%s" % ("PASS" if contract_pass else "FAIL"))


if __name__ == "__main__":
    main()

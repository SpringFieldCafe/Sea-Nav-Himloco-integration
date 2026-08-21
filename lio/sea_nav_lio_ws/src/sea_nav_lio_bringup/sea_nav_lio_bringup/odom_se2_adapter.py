#!/usr/bin/env python3
"""Convert Point-LIO planar pose into the SEA-Nav odom/base_link contract."""

import argparse
import math
from pathlib import Path


def load_base_to_lio(path):
    """Read the scalar calibration emitted by calibrate_lio_odom_se2.py."""
    values = {}
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        try:
            values[key.strip()] = float(raw_value.strip())
        except ValueError:
            continue
    keys = ("base_to_lio_x_m", "base_to_lio_y_m", "base_to_lio_yaw_rad")
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError("calibration YAML missing: " + ", ".join(missing))
    return tuple(values[key] for key in keys)


def invert_se2(base_to_lio):
    """Return T_lio_base, the inverse of the stored T_base_lio."""
    x, y, yaw = [float(value) for value in base_to_lio]
    c, s = math.cos(yaw), math.sin(yaw)
    return (-c * x - s * y, s * x - c * y, -yaw)


def compose_lio_base(lio_x, lio_y, lio_yaw, lio_to_base):
    """Compute T_odom_base = T_odom_lio * T_lio_base in SE(2)."""
    tx, ty, tyaw = [float(value) for value in lio_to_base]
    c, s = math.cos(float(lio_yaw)), math.sin(float(lio_yaw))
    return (
        float(lio_x) + c * tx - s * ty,
        float(lio_y) + s * tx + c * ty,
        float(lio_yaw) + tyaw,
    )


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def quaternion_from_yaw(yaw):
    from geometry_msgs.msg import Quaternion

    result = Quaternion()
    result.w = math.cos(float(yaw) / 2.0)
    result.z = math.sin(float(yaw) / 2.0)
    return result


class OdomSe2Adapter:
    def __init__(self, calibration_yaml, input_topic, output_topic):
        import rclpy
        from nav_msgs.msg import Odometry

        self.node = rclpy.create_node("sea_nav_lio_odom_se2_adapter")
        self.lio_to_base = invert_se2(load_base_to_lio(calibration_yaml))
        self.publisher = self.node.create_publisher(Odometry, output_topic, 10)
        self.node.create_subscription(Odometry, input_topic, self.callback, 10)

    def callback(self, msg):
        from nav_msgs.msg import Odometry

        lio_pose = msg.pose.pose
        lio_yaw = yaw_from_quaternion(lio_pose.orientation)
        x, y, yaw = compose_lio_base(
            lio_pose.position.x, lio_pose.position.y, lio_yaw, self.lio_to_base
        )
        output = Odometry()
        output.header = msg.header
        output.header.frame_id = "odom"
        output.child_frame_id = "base_link"
        output.pose.pose.position.x = x
        output.pose.pose.position.y = y
        output.pose.pose.position.z = 0.0
        output.pose.pose.orientation = quaternion_from_yaw(yaw)
        output.pose.covariance = msg.pose.covariance
        output.twist = msg.twist
        self.publisher.publish(output)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-yaml", required=True)
    parser.add_argument("--input-topic", default="/sea_nav/lio/odom")
    parser.add_argument("--output-topic", default="/sea_nav/lio/odom_base")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    import rclpy

    rclpy.init(args=None)
    adapter = OdomSe2Adapter(args.calibration_yaml, args.input_topic, args.output_topic)
    try:
        rclpy.spin(adapter.node)
    finally:
        adapter.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

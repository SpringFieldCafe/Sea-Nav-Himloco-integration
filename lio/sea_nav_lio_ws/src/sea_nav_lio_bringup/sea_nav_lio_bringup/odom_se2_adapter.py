#!/usr/bin/env python3
"""Convert Point-LIO IMU-sensor odometry into the odom/base_link contract.

The ROS entry point keeps its historical ``odom_se2_adapter`` name, but the
runtime path is a full SE(3) pose and twist transform. The old calibration
argument is accepted for CLI compatibility and is intentionally not used by
the native sensor-frame path.
"""

import argparse
import math
from pathlib import Path


# Unitree Go2 L1 mounting contract. T_lidar_to_imu is the Point-LIO
# convention p_imu = R_lidar_to_imu * p_lidar + T_lidar_to_imu.
LIDAR_TO_BASE_YAW = 2.8782
LIDAR_TO_BASE_TRANSLATION = (0.28945, 0.0, -0.046825)
LIDAR_TO_IMU_TRANSLATION = (0.007698, 0.014655, -0.00667)


def _vec_add(a, b):
    return tuple(float(a[i]) + float(b[i]) for i in range(3))


def _vec_neg(value):
    return tuple(-float(component) for component in value)


def _cross(a, b):
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _quat_normalize(q):
    norm = math.sqrt(sum(float(component) ** 2 for component in q))
    if norm <= 1e-12:
        raise ValueError("zero-norm quaternion")
    return tuple(float(component) / norm for component in q)


def _quat_conjugate(q):
    return (-float(q[0]), -float(q[1]), -float(q[2]), float(q[3]))


def _quat_multiply(a, b):
    ax, ay, az, aw = [float(value) for value in a]
    bx, by, bz, bw = [float(value) for value in b]
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_rotate(q, vector):
    q = _quat_normalize(q)
    pure = (float(vector[0]), float(vector[1]), float(vector[2]), 0.0)
    rotated = _quat_multiply(_quat_multiply(q, pure), _quat_conjugate(q))
    return rotated[:3]


def quaternion_from_rpy(roll, pitch, yaw):
    cr = math.cos(float(roll) / 2.0)
    sr = math.sin(float(roll) / 2.0)
    cp = math.cos(float(pitch) / 2.0)
    sp = math.sin(float(pitch) / 2.0)
    cy = math.cos(float(yaw) / 2.0)
    sy = math.sin(float(yaw) / 2.0)
    return _quat_normalize((
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ))


def invert_transform(rotation, translation):
    """Return the inverse of a transform mapping child coordinates to parent."""
    inverse_rotation = _quat_conjugate(_quat_normalize(rotation))
    inverse_translation = _quat_rotate(inverse_rotation, _vec_neg(translation))
    return inverse_rotation, inverse_translation


def compose_transform(first_rotation, first_translation,
                      second_rotation, second_translation):
    """Compose parent<-middle and middle<-child transforms."""
    rotation = _quat_multiply(first_rotation, second_rotation)
    translation = _vec_add(
        first_translation,
        _quat_rotate(first_rotation, second_translation),
    )
    return _quat_normalize(rotation), translation


def unitree_imu_to_base_transform():
    """Return T_imu_base for T_odom_base=T_odom_imu*T_imu_base.

    The returned transform maps base coordinates into IMU coordinates. It is
    derived as inverse(T_base_imu), where T_base_imu is composed from the
    official lidar-to-base mount and the inverse of Point-LIO's lidar-to-IMU
    coordinate transform.
    """
    lidar_to_base = (
        quaternion_from_rpy(0.0, LIDAR_TO_BASE_YAW, 0.0),
        LIDAR_TO_BASE_TRANSLATION,
    )
    lidar_to_imu = (
        (0.0, 0.0, 0.0, 1.0),
        LIDAR_TO_IMU_TRANSLATION,
    )
    imu_to_lidar = invert_transform(*lidar_to_imu)
    base_to_imu = compose_transform(*lidar_to_base, *imu_to_lidar)
    return invert_transform(*base_to_imu)


def transform_pose(imu_position, imu_orientation, imu_to_base=None):
    """Apply T_odom_base=T_odom_imu*T_imu_base to a pose."""
    if imu_to_base is None:
        imu_to_base = unitree_imu_to_base_transform()
    fixed_rotation, fixed_translation = imu_to_base
    output_orientation = _quat_normalize(
        _quat_multiply(imu_orientation, fixed_rotation)
    )
    output_position = _vec_add(
        imu_position,
        _quat_rotate(imu_orientation, fixed_translation),
    )
    return output_position, output_orientation


def transform_twist(linear_imu, angular_imu, imu_to_base=None):
    """Transform child-frame IMU twist to the base origin and base frame."""
    if imu_to_base is None:
        imu_to_base = unitree_imu_to_base_transform()
    fixed_rotation, fixed_translation = imu_to_base
    base_rotation = _quat_conjugate(_quat_normalize(fixed_rotation))
    linear_at_base_imu = _vec_add(
        linear_imu,
        _cross(angular_imu, fixed_translation),
    )
    return (
        _quat_rotate(base_rotation, linear_at_base_imu),
        _quat_rotate(base_rotation, angular_imu),
    )


# Legacy helpers remain import-compatible for existing offline tools/tests.
def load_base_to_lio(path):
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
    x, y, yaw = [float(value) for value in base_to_lio]
    c, s = math.cos(yaw), math.sin(yaw)
    return (-c * x - s * y, s * x - c * y, -yaw)


def compose_lio_base(lio_x, lio_y, lio_yaw, lio_to_base):
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


def _ros_quaternion_to_tuple(q):
    return (q.x, q.y, q.z, q.w)


def _tuple_to_ros_quaternion(values):
    from geometry_msgs.msg import Quaternion

    result = Quaternion()
    result.x, result.y, result.z, result.w = values
    return result


class OdomSe3Adapter:
    def __init__(self, calibration_yaml=None, input_topic="/sea_nav/lio/odom",
                 output_topic="/sea_nav/lio/odom_base"):
        import rclpy
        from nav_msgs.msg import Odometry

        # Kept as an accepted argument for existing launch/scripts. The
        # native sensor-frame path intentionally uses the fixed Unitree SE(3)
        # contract instead of the old planar calibration.
        self.legacy_calibration_yaml = calibration_yaml
        self.imu_to_base = unitree_imu_to_base_transform()
        self.node = rclpy.create_node("sea_nav_lio_odom_se2_adapter")
        self.publisher = self.node.create_publisher(Odometry, output_topic, 10)
        self.node.create_subscription(Odometry, input_topic, self.callback, 10)

    def callback(self, msg):
        from nav_msgs.msg import Odometry

        position, orientation = transform_pose(
            (msg.pose.pose.position.x, msg.pose.pose.position.y,
             msg.pose.pose.position.z),
            _ros_quaternion_to_tuple(msg.pose.pose.orientation),
            self.imu_to_base,
        )
        linear, angular = transform_twist(
            (msg.twist.twist.linear.x, msg.twist.twist.linear.y,
             msg.twist.twist.linear.z),
            (msg.twist.twist.angular.x, msg.twist.twist.angular.y,
             msg.twist.twist.angular.z),
            self.imu_to_base,
        )

        output = Odometry()
        output.header = msg.header
        output.header.frame_id = "odom"
        output.child_frame_id = "base_link"
        output.pose.pose.position.x = position[0]
        output.pose.pose.position.y = position[1]
        output.pose.pose.position.z = position[2]
        output.pose.pose.orientation = _tuple_to_ros_quaternion(orientation)
        output.pose.covariance = msg.pose.covariance
        output.twist.twist.linear.x = linear[0]
        output.twist.twist.linear.y = linear[1]
        output.twist.twist.linear.z = linear[2]
        output.twist.twist.angular.x = angular[0]
        output.twist.twist.angular.y = angular[1]
        output.twist.twist.angular.z = angular[2]
        output.twist.covariance = msg.twist.covariance
        self.publisher.publish(output)


# Preserve the historical import/class name while using the SE(3) path.
OdomSe2Adapter = OdomSe3Adapter


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-yaml",
        default=None,
        help="Deprecated compatibility option; native SE(3) uses Unitree fixed extrinsics.",
    )
    parser.add_argument("--input-topic", default="/sea_nav/lio/odom")
    parser.add_argument("--output-topic", default="/sea_nav/lio/odom_base")
    return parser


def main(argv=None):
    import rclpy
    from rclpy.utilities import remove_ros_args

    # A ROS 2 executable launched through launch_ros receives remapping
    # arguments after ``--ros-args``.  They belong to rclpy, not this
    # adapter's application parser.
    cli_args = remove_ros_args(args=argv)
    # ``console_scripts`` normally exposes ``odom_se2_adapter`` here, while
    # direct module/script launches may expose a Python file path instead.
    # In both cases the first non-option is the executable name, not an
    # adapter argument.
    if cli_args and not cli_args[0].startswith("-"):
        cli_args = cli_args[1:]
    args = build_parser().parse_args(cli_args)

    rclpy.init(args=None)
    adapter = OdomSe3Adapter(args.calibration_yaml, args.input_topic, args.output_topic)
    try:
        rclpy.spin(adapter.node)
    finally:
        adapter.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

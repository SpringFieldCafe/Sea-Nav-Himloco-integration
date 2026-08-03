"""ROS 2 read-only subscriptions for the Go2 onboard shadow runtime."""

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Latest:
    value: object = None
    received_at: float = 0.0

    @property
    def age(self):
        return None if self.value is None else time.monotonic() - self.received_at


@dataclass
class LowStateData:
    quaternion: np.ndarray
    gyro: np.ndarray
    q_motor: np.ndarray
    dq_motor: np.ndarray


@dataclass
class OdomData:
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    position: np.ndarray


class RosStateReader:
    """Subscribes only to sensor/state topics; never creates /lowcmd."""

    def __init__(self, topics):
        import rclpy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2
        from unitree_go.msg import LowState, WirelessController

        self.lowstate = Latest()
        self.lidar = Latest()
        self.odom = Latest()
        self.wireless = Latest()
        self.node = rclpy.create_node("sea_nav_go2_shadow_runtime")
        self.node.create_subscription(LowState, topics.lowstate, self._lowstate_callback, 10)
        self.node.create_subscription(PointCloud2, topics.lidar, self._lidar_callback, 10)
        self.node.create_subscription(Odometry, topics.odom, self._odom_callback, 10)
        self.node.create_subscription(WirelessController, topics.wireless, self._wireless_callback, 10)

    def _store(self, slot, value):
        slot.value = value
        slot.received_at = time.monotonic()

    def _lowstate_callback(self, msg):
        motors = getattr(msg, "motor_state", getattr(msg, "motor_states", None))
        if motors is None or len(motors) < 12:
            raise RuntimeError("LowState has fewer than 12 motor states")
        imu = msg.imu_state
        self._store(self.lowstate, LowStateData(
            quaternion=np.asarray(imu.quaternion, dtype=np.float32),
            gyro=np.asarray(imu.gyroscope, dtype=np.float32),
            q_motor=np.asarray([motor.q for motor in motors[:12]], dtype=np.float32),
            dq_motor=np.asarray([motor.dq for motor in motors[:12]], dtype=np.float32),
        ))

    def _lidar_callback(self, msg):
        from sensor_msgs_py import point_cloud2
        points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        self._store(self.lidar, np.asarray(points, dtype=np.float32).reshape((-1, 3)))

    def _odom_callback(self, msg):
        twist = msg.twist.twist
        position = msg.pose.pose.position
        self._store(self.odom, OdomData(
            linear_velocity=np.asarray([twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float32),
            angular_velocity=np.asarray([twist.angular.x, twist.angular.y, twist.angular.z], dtype=np.float32),
            position=np.asarray([position.x, position.y, position.z], dtype=np.float32),
        ))

    def _wireless_callback(self, msg):
        self._store(self.wireless, msg)

    def spin_once(self, timeout_sec=0.0):
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def close(self):
        self.node.destroy_node()

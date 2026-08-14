"""ROS 2 read-only subscriptions for the Go2 onboard shadow runtime."""

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Latest:
    value: object = None
    received_at: float = 0.0
    source_timestamp: float = 0.0
    frame_id: str = ""
    child_frame_id: str = ""
    count: int = 0
    frequency_hz: float = 0.0
    _previous_received_at: float = 0.0

    @property
    def age(self):
        return None if self.value is None else time.monotonic() - self.received_at

    def summary(self):
        return {
            "age_s": self.age,
            "frequency_hz": self.frequency_hz,
            "count": self.count,
            "frame_id": self.frame_id,
            "child_frame_id": self.child_frame_id,
            "source_timestamp": self.source_timestamp,
        }


@dataclass
class LowStateData:
    quaternion: np.ndarray
    gyro: np.ndarray
    q_motor: np.ndarray
    dq_motor: np.ndarray
    frame_id: str = ""


@dataclass
class OdomData:
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    position: np.ndarray
    frame_id: str = ""
    child_frame_id: str = ""


class RosStateReader:
    """Subscribes only to sensor/state topics; never creates /lowcmd."""

    def __init__(self, topics):
        import rclpy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2
        from unitree_go.msg import LowState, WirelessController
        from geometry_msgs.msg import PointStamped

        self.lowstate = Latest()
        self.lidar = Latest()
        self.odom = Latest()
        self.wireless = Latest()
        self.goal = Latest()
        self.node = rclpy.create_node("sea_nav_go2_shadow_runtime")
        self.node.create_subscription(LowState, topics.lowstate, self._lowstate_callback, 10)
        self.node.create_subscription(PointCloud2, topics.lidar, self._lidar_callback, 10)
        self.node.create_subscription(Odometry, topics.odom, self._odom_callback, 10)
        self.node.create_subscription(WirelessController, topics.wireless, self._wireless_callback, 10)
        if getattr(topics, "goal", ""):
            self.node.create_subscription(PointStamped, topics.goal, self._goal_callback, 10)

    def _store(self, slot, value, message=None):
        now = time.monotonic()
        if slot.received_at:
            period = now - slot.received_at
            if period > 0:
                slot.frequency_hz = 1.0 / period
        slot._previous_received_at = slot.received_at
        slot.value = value
        slot.received_at = now
        slot.count += 1
        slot.source_timestamp = _message_timestamp(message)
        slot.frame_id = _message_frame_id(message)
        slot.child_frame_id = str(getattr(message, "child_frame_id", "") or "")

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
            frame_id=_message_frame_id(msg),
        ), msg)

    def _lidar_callback(self, msg):
        from sensor_msgs_py import point_cloud2
        points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        self._store(self.lidar, np.asarray(points, dtype=np.float32).reshape((-1, 3)), msg)

    def _odom_callback(self, msg):
        twist = msg.twist.twist
        position = msg.pose.pose.position
        self._store(self.odom, OdomData(
            linear_velocity=np.asarray([twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float32),
            angular_velocity=np.asarray([twist.angular.x, twist.angular.y, twist.angular.z], dtype=np.float32),
            position=np.asarray([position.x, position.y, position.z], dtype=np.float32),
            frame_id=_message_frame_id(msg),
            child_frame_id=str(getattr(msg, "child_frame_id", "") or ""),
        ), msg)

    def _wireless_callback(self, msg):
        self._store(self.wireless, msg, msg)

    def _goal_callback(self, msg):
        from .goal import Goal2D
        point = msg.point
        self._store(self.goal, Goal2D(
            frame_id=_message_frame_id(msg),
            x=float(point.x), y=float(point.y), timestamp=_message_timestamp(msg),
        ), msg)

    def spin_once(self, timeout_sec=0.0):
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def close(self):
        self.node.destroy_node()

    def health(self):
        values = []
        for slot in (self.lowstate, self.lidar, self.odom, self.wireless, self.goal):
            if slot.value is not None:
                values.append(slot.value if not hasattr(slot.value, "__dict__") else _dataclass_values(slot.value))
        return {
            "ages": {
                "lowstate": self.lowstate.age,
                "lidar": self.lidar.age,
                "odom": self.odom.age,
                "wireless": self.wireless.age,
                "goal": self.goal.age,
            },
            "streams": {
                "lowstate": self.lowstate.summary(),
                "lidar": self.lidar.summary(),
                "odom": self.odom.summary(),
                "wireless": self.wireless.summary(),
                "goal": self.goal.summary(),
            },
            "finite_values": values,
            "wireless_emergency": wireless_emergency(self.wireless.value),
        }


def _message_timestamp(message):
    if message is None or not hasattr(message, "header"):
        return 0.0
    stamp = message.header.stamp
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


def _message_frame_id(message):
    if message is None:
        return ""
    return str(getattr(getattr(message, "header", None), "frame_id", "") or getattr(message, "frame_id", "") or "")


def _dataclass_values(value):
    return [getattr(value, name) for name in getattr(value, "__dataclass_fields__", {})]


def wireless_emergency(message):
    """Use only explicit boolean emergency fields; do not guess Unitree key bits."""
    if message is None:
        return False
    for name in ("emergency_stop", "estop", "e_stop"):
        if hasattr(message, name) and bool(getattr(message, name)):
            return True
    return False

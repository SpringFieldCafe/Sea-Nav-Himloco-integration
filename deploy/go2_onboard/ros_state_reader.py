"""ROS 2 read-only subscriptions for the Go2 onboard shadow runtime."""

import time
from dataclasses import dataclass
import threading

import numpy as np

from .lidar_ray_adapter import LidarRayCache


@dataclass
class Latest:
    value: object = None
    received_at: float = 0.0
    source_timestamp: float = 0.0
    frame_id: str = ""
    child_frame_id: str = ""
    count: int = 0
    frequency_hz: float = 0.0
    last_period_s: float = 0.0
    max_gap_s: float = 0.0
    gap_count: int = 0
    stale_count: int = 0
    message_type: str = ""
    point_count: int = 0
    finite_point_count: int = 0
    invalid_point_count: int = 0
    range_min_m: float = 0.0
    range_max_m: float = 0.0
    near_origin_point_count: int = 0
    _previous_received_at: float = 0.0
    _stale_reported: bool = False

    @property
    def age(self):
        return None if self.value is None else time.monotonic() - self.received_at

    def summary(self):
        return {
            "age_s": self.age,
            "frequency_hz": self.frequency_hz,
            "last_period_s": self.last_period_s,
            "max_gap_s": self.max_gap_s,
            "gap_count": self.gap_count,
            "stale_count": self.stale_count,
            "count": self.count,
            "message_type": self.message_type,
            "point_count": self.point_count,
            "finite_point_count": self.finite_point_count,
            "invalid_point_count": self.invalid_point_count,
            "finite_ratio": (
                self.finite_point_count / self.point_count if self.point_count else 0.0
            ),
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
            "near_origin_point_count": self.near_origin_point_count,
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
    orientation_quaternion: np.ndarray = None
    frame_id: str = ""
    child_frame_id: str = ""


class RosStateReader:
    """Subscribes only to sensor/state topics; never creates /lowcmd."""

    def __init__(self, topics, max_sensor_age=0.25):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2
        from unitree_go.msg import LowState, WirelessController
        from geometry_msgs.msg import PointStamped

        self._rclpy = rclpy
        self._owns_rclpy = False
        self.lock = threading.RLock()
        self._executor = None
        self._spin_thread = None
        self.max_sensor_age = float(max_sensor_age)
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        self.lowstate = Latest()
        self.lidar = Latest()
        self.lidar_cache = LidarRayCache()
        self.odom = Latest()
        self.wireless = Latest()
        self.goal = Latest()
        self.node = rclpy.create_node("sea_nav_go2_shadow_runtime")
        self._executor_type = SingleThreadedExecutor
        self.node.create_subscription(LowState, topics.lowstate, self._lowstate_callback, 10)
        self.node.create_subscription(PointCloud2, topics.lidar, self._lidar_callback, 10)
        self.node.create_subscription(Odometry, topics.odom, self._odom_callback, 10)
        self.node.create_subscription(WirelessController, topics.wireless, self._wireless_callback, 10)
        if getattr(topics, "goal", ""):
            self.node.create_subscription(PointStamped, topics.goal, self._goal_callback, 10)

    def _store(self, slot, value, message=None):
        with self.lock:
            now = time.monotonic()
            if slot.received_at:
                period = now - slot.received_at
                if period > 0:
                    slot.last_period_s = period
                    slot.max_gap_s = max(slot.max_gap_s, period)
                    if period > self.max_sensor_age:
                        slot.gap_count += 1
                    slot.frequency_hz = 1.0 / period
            slot._previous_received_at = slot.received_at
            slot.value = value
            slot.received_at = now
            slot.count += 1
            slot.source_timestamp = _message_timestamp(message)
            slot.frame_id = _message_frame_id(message)
            slot.child_frame_id = str(getattr(message, "child_frame_id", "") or "")
            if message is not None:
                slot.message_type = f"{type(message).__module__}.{type(message).__name__}"

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
        points = np.asarray(
            list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False))
        )
        if points.dtype.names:
            raw = np.column_stack((points["x"], points["y"], points["z"])).astype(
                np.float32, copy=False
            )
        else:
            raw = np.asarray(points, dtype=np.float32).reshape((-1, 3))
        finite = np.isfinite(raw).all(axis=1)
        points = raw[finite]
        received_at = time.monotonic()
        self.lidar_cache.update(
            points,
            received_at=received_at,
            source_timestamp=_message_timestamp(msg),
        )
        with self.lock:
            self._store(self.lidar, points, msg)
            self.lidar.point_count = int(raw.shape[0])
            self.lidar.finite_point_count = int(finite.sum())
            self.lidar.invalid_point_count = int((~finite).sum())
            if points.size:
                ranges = np.linalg.norm(points[:, :2], axis=1)
                self.lidar.range_min_m = float(ranges.min())
                self.lidar.range_max_m = float(ranges.max())
                self.lidar.near_origin_point_count = int((ranges < 0.15).sum())
            else:
                self.lidar.range_min_m = 0.0
                self.lidar.range_max_m = 0.0
                self.lidar.near_origin_point_count = 0

    def _odom_callback(self, msg):
        twist = msg.twist.twist
        position = msg.pose.pose.position
        self._store(self.odom, OdomData(
            linear_velocity=np.asarray([twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float32),
            angular_velocity=np.asarray([twist.angular.x, twist.angular.y, twist.angular.z], dtype=np.float32),
            position=np.asarray([position.x, position.y, position.z], dtype=np.float32),
            orientation_quaternion=np.asarray([
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ], dtype=np.float32),
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
        if self._executor is not None:
            raise RuntimeError("spin_once cannot be used while background spinning is active")
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def start_background_spin(self):
        """Continuously service ROS callbacks while callers read latest snapshots."""
        if self._executor is not None:
            return
        self._executor = self._executor_type()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="go2-ros-callbacks",
            daemon=True,
        )
        self._spin_thread.start()

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=2.0)
            self._executor = None
            self._spin_thread = None
        self.node.destroy_node()
        if self._owns_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()

    def health(self, max_sensor_age=0.25, sensor_max_ages=None):
        thresholds = _resolve_sensor_max_ages(max_sensor_age, sensor_max_ages)
        with self.lock:
            slots = {
                "lowstate": self.lowstate,
                "lidar": self.lidar,
                "odom": self.odom,
                "wireless": self.wireless,
                "goal": self.goal,
            }
            for name, slot in slots.items():
                age = slot.age
                stale = not is_fresh(age, thresholds[name])
                if stale and not slot._stale_reported:
                    slot.stale_count += 1
                slot._stale_reported = stale
            values = []
            for slot in slots.values():
                if slot.value is not None:
                    values.append(slot.value if not hasattr(slot.value, "__dict__") else _dataclass_values(slot.value))
            ages = {name: slot.age for name, slot in slots.items()}
            return {
                "ages": ages,
                "sensor_ages": {name: ages[name] for name in ("lowstate", "lidar", "odom", "wireless")},
                "sensor_thresholds": thresholds,
                "streams": {name: slot.summary() for name, slot in slots.items()},
                "finite_values": values,
                "wireless_emergency": wireless_emergency(self.wireless.value),
            }


def _message_timestamp(message):
    if message is None or not hasattr(message, "header"):
        return 0.0
    stamp = message.header.stamp
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


def is_fresh(age, max_age):
    """Return whether a received sample is usable under its stream-specific budget."""
    if age is None:
        return False
    age = float(age)
    return np.isfinite(age) and 0.0 <= age <= float(max_age)


def _resolve_sensor_max_ages(default, overrides=None):
    values = {name: float(default) for name in ("lowstate", "odom", "lidar", "wireless", "goal")}
    if overrides:
        for name, value in overrides.items():
            if name in values:
                values[name] = float(value)
    if any(value <= 0.0 or not np.isfinite(value) for value in values.values()):
        raise ValueError("sensor freshness thresholds must be finite and positive")
    return values


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

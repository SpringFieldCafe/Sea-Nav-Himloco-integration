"""Read-only Go2 field diagnostics.

This tool subscribes to sensor/state topics only.  It deliberately has no
publisher, LowCmd import, sport-mode call, or motor write path.  Its purpose
is to collect the facts that cannot be established from the repository alone:
message types, measured timing, frame IDs, odometry fields, LiDAR validity,
and raw WirelessController fields.
"""

import argparse
import math
import time

import numpy as np

from .goal import GoalManager
from .logger import JsonlLogger
from .ros_state_reader import RosStateReader


class Topics:
    def __init__(self, args):
        self.lowstate = args.lowstate_topic
        self.lidar = args.lidar_topic
        self.odom = args.odom_topic
        self.wireless = args.wireless_topic
        self.goal = args.goal_topic


def quat_to_yaw(quaternion):
    q = np.asarray(quaternion, dtype=np.float32).reshape(-1)
    if q.size != 4 or not np.isfinite(q).all():
        raise ValueError("IMU quaternion must be finite [w,x,y,z]")
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _ros_value(value, depth=0):
    """Convert a ROS message to bounded JSON-safe diagnostic data."""
    if depth > 3:
        return "<depth-limit>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_ros_value(item, depth + 1) for item in value[:64]]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    field_names = []
    getter = getattr(value, "get_fields_and_field_types", None)
    if callable(getter):
        field_names = list(getter())
    else:
        field_names = [name for name in getattr(value, "__slots__", ()) if isinstance(name, str)]
        field_names = [name.lstrip("_") for name in field_names]
    if field_names:
        output = {}
        for name in field_names:
            try:
                output[name] = _ros_value(getattr(value, name), depth + 1)
            except (AttributeError, TypeError, ValueError):
                output[name] = "<unreadable>"
        return output
    return str(value)


def _odom_record(reader):
    odom = reader.odom.value
    if odom is None:
        return None
    return {
        "header_frame_id": odom.frame_id,
        "child_frame_id": odom.child_frame_id,
        "position_xyz_m": odom.position.tolist(),
        "orientation_wxyz": (
            odom.orientation_quaternion.tolist()
            if odom.orientation_quaternion is not None else None
        ),
        "twist_linear_raw": odom.linear_velocity.tolist(),
        "twist_angular_raw": odom.angular_velocity.tolist(),
        "twist_frame_semantics": "UNVERIFIED: compare against child_frame_id and field experiment",
    }


class ReadOnlyDiagnostics:
    def __init__(self, args):
        self.args = args
        self.reader = RosStateReader(Topics(args), max_sensor_age=args.max_sensor_age)
        self.logger = JsonlLogger(args.log)
        self.goal_manager = GoalManager()
        from .lidar_ray_adapter import NumpyLidarRayAdapter
        self.ray_adapter = NumpyLidarRayAdapter(
            ray_count=41, min_distance=0.1, max_distance=5.0,
            angle_min=-2.0 * np.pi / 3.0, angle_max=2.0 * np.pi / 3.0,
            min_z=-0.25, max_z=1.0,
        )
        self.started = time.monotonic()
        self.last_report = 0.0

    def run(self):
        print("[diagnostics] read-only Go2 field diagnostics started")
        print("[safety] LOWCMD DISABLED: subscriptions only; no command publisher")
        try:
            while self.args.duration <= 0 or time.monotonic() - self.started < self.args.duration:
                self.reader.spin_once(0.1)
                if self.reader.goal.value is not None:
                    self.goal_manager.update(self.reader.goal.value)
                now = time.monotonic()
                if now - self.last_report >= self.args.report_interval:
                    self.last_report = now
                    self.logger.write(self.record())
        except KeyboardInterrupt:
            print("[diagnostics] stopped")
        finally:
            self.logger.close()
            self.reader.close()

    def record(self):
        health = self.reader.health(self.args.max_sensor_age)
        low = self.reader.lowstate.value
        goal_robot_xy = None
        if low is not None and self.reader.odom.value is not None and self.goal_manager.current is not None:
            odom = self.reader.odom.value
            goal_robot_xy = self.goal_manager.relative_xy(
                odom.position[:2], quat_to_yaw(low.quaternion), odom.frame_id, self.args.base_frame,
            ).tolist()

        lidar = self.reader.lidar.value
        lidar_record = dict(health["streams"]["lidar"])
        if lidar is not None:
            rays = self.ray_adapter.project(lidar)[0]
            log2_rays = np.log2(np.clip(rays, 0.1, 5.0))
            lidar_record.update({
                "points_used_by_adapter": int(lidar.shape[0]),
                "rays_41": rays.tolist(),
                "rays_min_m": float(rays.min()),
                "rays_max_m": float(rays.max()),
                "log2_rays_min": float(log2_rays.min()),
                "log2_rays_max": float(log2_rays.max()),
                "ray_contract": {
                    "count": 41, "clip_m": [0.1, 5.0],
                    "angle_rad": [-2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0],
                },
            })

        return {
            "mode": "field_diagnostics",
            "timestamp": time.time(),
            "runtime_state": "READ_ONLY",
            "sensor": health,
            "odom": _odom_record(self.reader),
            "goal": self.goal_manager.current,
            "goal_robot_xy": goal_robot_xy,
            "wireless_raw_fields": _ros_value(self.reader.wireless.value),
            "lidar": lidar_record,
            "lowstate_shapes": _lowstate_shapes(low),
            "lowcmd_sent": False,
        }


def _lowstate_shapes(low):
    if low is None:
        return None
    return {
        "quaternion_shape": list(low.quaternion.shape),
        "gyro_shape": list(low.gyro.shape),
        "q_motor_shape": list(low.q_motor.shape),
        "dq_motor_shape": list(low.dq_motor.shape),
        "q_motor_finite": bool(np.isfinite(low.q_motor).all()),
        "dq_motor_finite": bool(np.isfinite(low.dq_motor).all()),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Read-only Unitree Go2 field diagnostics")
    parser.add_argument("--lowstate-topic", default="/lowstate")
    parser.add_argument("--lidar-topic", default="/utlidar/cloud_base")
    parser.add_argument("--odom-topic", default="/utlidar/robot_odom")
    parser.add_argument("--wireless-topic", default="/wirelesscontroller")
    parser.add_argument("--goal-topic", default="/sea_nav/goal2d")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--report-interval", type=float, default=1.0)
    parser.add_argument("--max-sensor-age", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--log", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.report_interval <= 0 or args.max_sensor_age <= 0:
        raise SystemExit("--report-interval and --max-sensor-age must be positive")
    ReadOnlyDiagnostics(args).run()


if __name__ == "__main__":
    main()

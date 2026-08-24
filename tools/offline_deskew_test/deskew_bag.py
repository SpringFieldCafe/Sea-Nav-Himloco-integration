#!/usr/bin/env python3
"""Offline-only PointCloud2 deskew diagnostic.

The estimator is deliberately not touched.  This tool integrates the
transformed IMU stream once, then rewrites each cloud into its scan-end frame:

    p_end = T_end^-1 * T_point * p_point

The integration is intended for controlled A/B diagnosis, not deployment.
"""

from __future__ import annotations

import argparse
import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time as RosTime
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


def stamp_to_sec(stamp: RosTime) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def sec_to_stamp(value: float) -> RosTime:
    sec = math.floor(value)
    nanosec = int(round((value - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    result = RosTime()
    result.sec = int(sec)
    result.nanosec = nanosec
    return result


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2]])
    axis = rotvec / angle
    half = 0.5 * angle
    return np.array([math.cos(half), *(math.sin(half) * axis)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class ImuSample:
    t: float
    gyro: np.ndarray
    accel: np.ndarray


@dataclass
class Pose:
    t: float
    rotation: np.ndarray
    position: np.ndarray


BagPayload = Union[bytes, PointCloud2]


def read_bag(path: Path) -> Tuple[List[Tuple[str, BagPayload, int]], List[ImuSample], List[Tuple[str, PointCloud2, int]], List[rosbag2_py.TopicMetadata]]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", "cdr"),
    )
    topics = list(reader.get_all_topics_and_types())
    message_records: List[Tuple[str, BagPayload, int]] = []
    imu_samples: List[ImuSample] = []
    clouds: List[Tuple[str, PointCloud2, int]] = []
    while reader.has_next():
        topic, data, bag_time = reader.read_next()
        if topic == "/sea_nav/lio/transformed_raw_imu":
            msg = deserialize_message(data, Imu)
            imu_samples.append(ImuSample(
                stamp_to_sec(msg.header.stamp),
                np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=float),
                np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=float),
            ))
            message_records.append((topic, data, bag_time))
        elif topic == "/sea_nav/lio/transformed_cloud":
            cloud = deserialize_message(data, PointCloud2)
            clouds.append((topic, cloud, bag_time))
            message_records.append((topic, cloud, bag_time))
        else:
            message_records.append((topic, data, bag_time))
    imu_samples.sort(key=lambda sample: sample.t)
    clouds.sort(key=lambda record: stamp_to_sec(record[1].header.stamp))
    return message_records, imu_samples, clouds, topics


def integrate_poses(samples: Sequence[ImuSample]) -> List[Pose]:
    if not samples:
        raise ValueError("bag has no transformed_raw_imu samples")
    # A stationary bag supplies a stable specific-force vector. Subtracting
    # its mean avoids inventing translation during this diagnostic replay.
    gravity_sample = np.mean([sample.accel for sample in samples], axis=0)
    poses = [Pose(samples[0].t, np.eye(3), np.zeros(3))]
    q = np.array([1.0, 0.0, 0.0, 0.0])
    position = np.zeros(3)
    velocity = np.zeros(3)
    for previous, current in zip(samples, samples[1:]):
        dt = current.t - previous.t
        if dt <= 0 or dt > 0.1:
            poses.append(Pose(current.t, quat_to_rot(q), position.copy()))
            continue
        gyro = 0.5 * (previous.gyro + current.gyro)
        accel = 0.5 * (previous.accel + current.accel) - gravity_sample
        q = quat_normalize(quat_mul(q, quat_from_rotvec(gyro * dt)))
        rotation = quat_to_rot(q)
        acceleration_world = rotation @ accel
        position = position + velocity * dt + 0.5 * acceleration_world * dt * dt
        velocity = velocity + acceleration_world * dt
        poses.append(Pose(current.t, rotation, position.copy()))
    return poses


def interpolate_pose(poses: Sequence[Pose], t: float) -> Pose:
    if t <= poses[0].t:
        return poses[0]
    if t >= poses[-1].t:
        return poses[-1]
    index = max(0, min(len(poses) - 2, np.searchsorted([pose.t for pose in poses], t) - 1))
    first, second = poses[index], poses[index + 1]
    alpha = (t - first.t) / max(second.t - first.t, 1e-12)
    # Linear interpolation is sufficient for this diagnostic's small IMU dt.
    rotation = first.rotation @ second.rotation.T
    del rotation  # keep the interpolation explicitly position-only below
    position = (1 - alpha) * first.position + alpha * second.position
    return Pose(t, first.rotation, position)


def pose_at(poses: Sequence[Pose], t: float, times: Sequence[float] | None = None) -> Pose:
    if t <= poses[0].t:
        return poses[0]
    if t >= poses[-1].t:
        return poses[-1]
    if times is None:
        times = [pose.t for pose in poses]
    index = max(0, min(len(poses) - 2, bisect.bisect_right(times, t) - 1))
    first, second = poses[index], poses[index + 1]
    alpha = (t - first.t) / max(second.t - first.t, 1e-12)
    # Interpolate orientation through the relative rotation's axis-angle.
    relative = first.rotation.T @ second.rotation
    trace = max(-1.0, min(3.0, float(np.trace(relative))))
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) * 0.5)))
    if angle < 1e-9:
        interpolated = first.rotation
    else:
        axis = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0], relative[1, 0] - relative[0, 1]]) / (2 * math.sin(angle))
        interpolated = first.rotation @ quat_to_rot(quat_from_rotvec(axis * angle * alpha))
    position = (1 - alpha) * first.position + alpha * second.position
    return Pose(t, interpolated, position)


def deskew_cloud(cloud: PointCloud2, poses: Sequence[Pose]) -> PointCloud2:
    fields = {field.name for field in cloud.fields}
    if "time" not in fields:
        raise ValueError("cloud has no time field")
    points = list(pc2.read_points(cloud, field_names=None, skip_nans=False))
    names = [field.name for field in cloud.fields]
    x_index, y_index, z_index = names.index("x"), names.index("y"), names.index("z")
    time_index = names.index("time")
    start = stamp_to_sec(cloud.header.stamp)
    times = [float(point[time_index]) for point in points if math.isfinite(float(point[time_index]))]
    if not times:
        raise ValueError("cloud time field contains no finite values")
    end_time = start + max(times)
    pose_times = [pose.t for pose in poses]
    end_pose = pose_at(poses, end_time, pose_times)
    corrected = []
    for point in points:
        values = list(point)
        point_time = float(values[time_index])
        if not math.isfinite(point_time):
            corrected.append(tuple(values))
            continue
        point_pose = pose_at(poses, start + point_time, pose_times)
        point_world = point_pose.rotation @ np.array([float(values[x_index]), float(values[y_index]), float(values[z_index])]) + point_pose.position
        point_end = end_pose.rotation.T @ (point_world - end_pose.position)
        values[x_index], values[y_index], values[z_index] = map(float, point_end)
        corrected.append(tuple(values))
    output = pc2.create_cloud(cloud.header, cloud.fields, corrected)
    output.is_dense = cloud.is_dense
    return output


def write_bag(output: Path, records: Sequence[Tuple[str, BagPayload, int]], clouds: Sequence[Tuple[str, PointCloud2, int]], topics: Sequence[rosbag2_py.TopicMetadata], poses: Sequence[Pose]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(output), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    for topic in topics:
        writer.create_topic(topic)
    for topic, payload, bag_time in sorted(records, key=lambda value: value[2]):
        if isinstance(payload, PointCloud2):
            payload = serialize_message(deskew_cloud(payload, poses))
        writer.write(topic, payload, bag_time)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bag", type=Path, required=True)
    parser.add_argument("--output-bag", type=Path, required=True)
    args = parser.parse_args()
    records, imu_samples, clouds, topics = read_bag(args.input_bag)
    poses = integrate_poses(imu_samples)
    write_bag(args.output_bag, records, clouds, topics, poses)
    spans = []
    for _, cloud, _ in clouds:
        times = [float(point[-1]) for point in pc2.read_points(cloud, field_names=["time"], skip_nans=True)]
        if times:
            spans.append(max(times) - min(times))
    print(f"DESKEW_BAG_WRITTEN={args.output_bag}")
    print(f"CLOUD_FRAMES={len(clouds)}")
    print(f"IMU_SAMPLES={len(imu_samples)}")
    print(f"POINT_TIME_SPAN_MEAN={np.mean(spans):.9f}")
    print("DESKEW_REFERENCE=SCAN_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

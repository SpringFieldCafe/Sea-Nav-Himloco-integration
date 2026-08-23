#!/usr/bin/env python3
"""Standalone ROS 2 LiDAR deskew diagnostic node.

This node is intentionally not referenced by any existing launch file. It
rewrites only PointCloud2 xyz values and preserves the original fields.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from threading import Lock

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2, PointField


@dataclass
class ImuSample:
    stamp: float
    gyro: np.ndarray
    accel: np.ndarray


@dataclass
class CloudWorkItem:
    message: PointCloud2
    frame_id: int
    received_at: float
    queue_before: int
    queue_after: int


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_from_rotvec(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.array([1.0, vector[0] * 0.5, vector[1] * 0.5, vector[2] * 0.5])
    axis = vector / angle
    half = angle * 0.5
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


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class LidarDeskewNode(Node):
    def __init__(self) -> None:
        super().__init__("sea_nav_lidar_deskew")
        self.declare_parameter("imu_buffer_seconds", 2.0)
        self.declare_parameter("diagnostic_period_seconds", 1.0)
        # 1 preserves every input point. Values >1 are an explicit diagnostic
        # mode and keep every PointCloud2 field while selecting every Nth point.
        self.declare_parameter("point_downsample", 1)
        self.imu_buffer = deque()
        self.cloud_queue = queue.Queue(maxsize=2)
        self.buffer_lock = threading.Condition(Lock())
        self.worker_stop = threading.Event()
        self.worker = threading.Thread(target=self.worker_loop, name="deskew-worker", daemon=True)
        self.imu_group = ReentrantCallbackGroup()
        self.cloud_group = ReentrantCallbackGroup()
        self.gravity_samples = deque(maxlen=500)
        self.last_diag = self.get_clock().now().nanoseconds * 1e-9
        self.diag_lock = Lock()
        self.diag_imu_count = 0
        self.diag_cloud_count = 0
        self.diag_processed_count = 0
        self.diag_process_time_sum = 0.0
        self.diag_frame_id = 0
        self.diag_drop_count = 0

        self.imu_sub = self.create_subscription(
            Imu, "/sea_nav/lio/transformed_raw_imu", self.imu_callback, 100,
            callback_group=self.imu_group,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2, "/sea_nav/lio/transformed_cloud", self.cloud_callback, 20,
            callback_group=self.cloud_group,
        )
        self.cloud_pub = self.create_publisher(
            PointCloud2, "/sea_nav/lio/deskewed_cloud", 20
        )
        self.status_timer = self.create_timer(1.0, self.publish_status)
        self.worker.start()

    def imu_callback(self, message: Imu) -> None:
        sample = ImuSample(
            stamp_seconds(message.header.stamp),
            np.array([message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z], dtype=float),
            np.array([message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z], dtype=float),
        )
        with self.buffer_lock:
            self.imu_buffer.append(sample)
            self.gravity_samples.append(sample.accel)
            cutoff = sample.stamp - float(self.get_parameter("imu_buffer_seconds").value)
            while self.imu_buffer and self.imu_buffer[0].stamp < cutoff:
                self.imu_buffer.popleft()
            self.buffer_lock.notify_all()
        with self.diag_lock:
            self.diag_imu_count += 1

    def snapshot_imu(self, start: float, end: float):
        with self.buffer_lock:
            samples = list(self.imu_buffer)
            gravity = np.mean(list(self.gravity_samples), axis=0) if self.gravity_samples else np.array([0.0, 0.0, 9.81])
        usable = [sample for sample in samples if start - 0.02 <= sample.stamp <= end + 0.02]
        return usable, gravity

    def worker_loop(self) -> None:
        while not self.worker_stop.is_set():
            try:
                work = self.cloud_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                message = work.message
                worker_start = time.monotonic()
                decode_start = time.perf_counter()
                cloud = self._read_cloud_fields(message)
                decode_ms = (time.perf_counter() - decode_start) * 1000.0
                if cloud is None:
                    continue
                times = cloud["times"]
                scan_end = stamp_seconds(message.header.stamp) + float(np.max(times))
                wait_start = time.monotonic()
                deadline = time.monotonic() + 1.0
                while not self.worker_stop.is_set():
                    with self.buffer_lock:
                        latest = self.imu_buffer[-1].stamp if self.imu_buffer else None
                        if latest is not None and latest >= scan_end:
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self.buffer_lock.wait(timeout=min(remaining, 0.05))
                wait_end = time.monotonic()
                with self.buffer_lock:
                    imu_buffer_size = len(self.imu_buffer)
                self._process_cloud(
                    message,
                    work,
                    cloud,
                    worker_start=worker_start,
                    decode_ms=decode_ms,
                    wait_imu_ms=(wait_end - wait_start) * 1000.0,
                    imu_buffer_size=imu_buffer_size,
                )
            except Exception as error:
                self.get_logger().error(
                    "[DESKEW_EXCEPTION] "
                    f"frame_id={work.frame_id} "
                    f"exception={type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                self.cloud_queue.task_done()

    @staticmethod
    def integrate(samples, start, end, gravity):
        if len(samples) < 2:
            return None
        samples = sorted(samples, key=lambda sample: sample.stamp)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        position = np.zeros(3)
        velocity = np.zeros(3)
        poses = [(samples[0].stamp, q.copy(), position.copy())]
        for previous, current in zip(samples, samples[1:]):
            dt = current.stamp - previous.stamp
            if dt <= 0.0 or dt > 0.1:
                continue
            gyro = 0.5 * (previous.gyro + current.gyro)
            accel = 0.5 * (previous.accel + current.accel) - gravity
            q = quat_mul(q, quat_from_rotvec(gyro * dt))
            q /= max(float(np.linalg.norm(q)), 1e-12)
            rotation = quat_to_rotation(q)
            acceleration_world = rotation @ accel
            position += velocity * dt + 0.5 * acceleration_world * dt * dt
            velocity += acceleration_world * dt
            poses.append((current.stamp, q.copy(), position.copy()))
        if len(poses) < 2:
            return None
        return poses

    @staticmethod
    def pose_at(poses, stamp):
        if stamp <= poses[0][0]:
            return poses[0][1], poses[0][2]
        if stamp >= poses[-1][0]:
            return poses[-1][1], poses[-1][2]
        for first, second in zip(poses, poses[1:]):
            if first[0] <= stamp <= second[0]:
                ratio = (stamp - first[0]) / max(second[0] - first[0], 1e-12)
                q = first[1]  # small-dt diagnostic path; gyro integration dominates
                position = (1.0 - ratio) * first[2] + ratio * second[2]
                return q, position
        return poses[-1][1], poses[-1][2]

    def cloud_callback(self, message: PointCloud2) -> None:
        received_at = time.monotonic()
        with self.diag_lock:
            self.diag_cloud_count += 1
            self.diag_frame_id += 1
            frame_id = self.diag_frame_id
            drop_count = self.diag_drop_count
        queue_before = self.cloud_queue.qsize()
        work = CloudWorkItem(
            message=message,
            frame_id=frame_id,
            received_at=received_at,
            queue_before=queue_before,
            queue_after=queue_before,
        )
        try:
            self.cloud_queue.put_nowait(work)
            work.queue_after = self.cloud_queue.qsize()
        except queue.Full:
            try:
                self.cloud_queue.get_nowait()
                self.cloud_queue.task_done()
            except queue.Empty:
                pass
            with self.diag_lock:
                self.diag_drop_count += 1
                drop_count = self.diag_drop_count
            try:
                self.cloud_queue.put_nowait(work)
                work.queue_after = self.cloud_queue.qsize()
            except queue.Full:
                self.get_logger().warning("DESKEW_DROP cloud_queue_full")

    @staticmethod
    def _numpy_dtype(datatype: int, big_endian: bool):
        base_types = {
            PointField.INT8: np.int8,
            PointField.UINT8: np.uint8,
            PointField.INT16: np.int16,
            PointField.UINT16: np.uint16,
            PointField.INT32: np.int32,
            PointField.UINT32: np.uint32,
            PointField.FLOAT32: np.float32,
            PointField.FLOAT64: np.float64,
        }
        if datatype not in base_types:
            raise ValueError(f"unsupported PointCloud2 datatype={datatype}")
        dtype = np.dtype(base_types[datatype])
        return dtype.newbyteorder(">" if big_endian else "<")

    @classmethod
    def _field_view(cls, data, message: PointCloud2, field):
        if field.count != 1:
            raise ValueError(f"field {field.name} count={field.count} is unsupported")
        dtype = cls._numpy_dtype(field.datatype, message.is_bigendian)
        return np.ndarray(
            shape=(message.height, message.width),
            dtype=dtype,
            buffer=data,
            offset=field.offset,
            strides=(message.row_step, message.point_step),
        )

    def _read_cloud_fields(self, message: PointCloud2):
        fields = {field.name: field for field in message.fields}
        required = {"x", "y", "z", "time"}
        missing = required - fields.keys()
        if missing:
            self.get_logger().error(
                f"DESKEW_SKIP missing required fields={sorted(missing)}"
            )
            return None
        try:
            data = memoryview(message.data)
            views = {name: self._field_view(data, message, fields[name]) for name in required}
        except (TypeError, ValueError, BufferError) as error:
            self.get_logger().error(f"DESKEW_SKIP invalid PointCloud2 layout: {error}")
            return None
        times = np.asarray(views["time"], dtype=float).reshape(-1)
        downsample = int(self.get_parameter("point_downsample").value)
        indices = None
        if downsample > 1:
            indices = np.arange(times.size, dtype=np.intp)[::downsample]
            times = times[indices]
        finite = np.isfinite(times)
        if not np.any(finite):
            self.get_logger().error("DESKEW_SKIP no finite point time")
            return None
        return {
            # Keep the native PointCloud2 scalar views.  The arithmetic below
            # promotes them as needed, avoiding three full cloud copies here.
            "x": views["x"].reshape(-1) if indices is None else views["x"].reshape(-1)[indices],
            "y": views["y"].reshape(-1) if indices is None else views["y"].reshape(-1)[indices],
            "z": views["z"].reshape(-1) if indices is None else views["z"].reshape(-1)[indices],
            "times": times,
            "finite_time": finite,
            "indices": indices,
        }

    def _process_cloud(
        self,
        message: PointCloud2,
        work: CloudWorkItem,
        cloud,
        worker_start: float,
        decode_ms: float,
        wait_imu_ms: float,
        imu_buffer_size: int,
    ) -> None:
        process_start = time.perf_counter()
        times = cloud["times"]
        scan_start = stamp_seconds(message.header.stamp)
        scan_end = scan_start + float(np.max(times))
        snapshot_start = time.perf_counter()
        samples, gravity = self.snapshot_imu(scan_start, scan_end)
        snapshot_ms = (time.perf_counter() - snapshot_start) * 1000.0
        math_start = time.perf_counter()
        poses = self.integrate(samples, scan_start, scan_end, gravity)
        if poses is None:
            self.get_logger().warning(f"DESKEW_SKIP insufficient IMU samples count={len(samples)}")
            return
        q_end, p_end = self.pose_at(poses, scan_end)
        pose_times = np.asarray([pose[0] for pose in poses], dtype=float)
        pose_positions = np.asarray([pose[2] for pose in poses], dtype=float)
        r_end = quat_to_rotation(q_end)
        pose_quaternions = np.asarray([pose[1] for pose in poses], dtype=float)
        qw, qx, qy, qz = pose_quaternions.T
        pose_rot_flat = np.column_stack((
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qx * qw),
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx * qx + qy * qy),
        ))
        point_stamps = scan_start + times
        pose_indices = np.searchsorted(pose_times, point_stamps, side="right") - 1
        pose_indices = np.clip(pose_indices, 0, len(pose_times) - 2)
        next_indices = pose_indices + 1
        alpha = ((point_stamps - pose_times[pose_indices]) /
                 np.maximum(pose_times[next_indices] - pose_times[pose_indices], 1e-12))
        point_rot_flat = (
            pose_rot_flat[pose_indices] +
            alpha[:, None] * (pose_rot_flat[next_indices] - pose_rot_flat[pose_indices])
        )
        point_positions = (
            pose_positions[pose_indices] +
            alpha[:, None] * (pose_positions[next_indices] - pose_positions[pose_indices])
        )
        deskew_math_ms = (time.perf_counter() - math_start) * 1000.0
        xyz_start = time.perf_counter()
        x = cloud["x"]
        y = cloud["y"]
        z = cloud["z"]
        point_world = np.empty((times.shape[0], 3), dtype=float)
        point_world[:, 0] = (
            point_rot_flat[:, 0] * x + point_rot_flat[:, 1] * y + point_rot_flat[:, 2] * z
        )
        point_world[:, 1] = (
            point_rot_flat[:, 3] * x + point_rot_flat[:, 4] * y + point_rot_flat[:, 5] * z
        )
        point_world[:, 2] = (
            point_rot_flat[:, 6] * x + point_rot_flat[:, 7] * y + point_rot_flat[:, 8] * z
        )
        point_world += point_positions
        point_world -= p_end
        corrected_xyz = np.empty_like(point_world)
        corrected_xyz[:, 0] = (
            point_world[:, 0] * r_end[0, 0] + point_world[:, 1] * r_end[1, 0] + point_world[:, 2] * r_end[2, 0]
        )
        corrected_xyz[:, 1] = (
            point_world[:, 0] * r_end[0, 1] + point_world[:, 1] * r_end[1, 1] + point_world[:, 2] * r_end[2, 1]
        )
        corrected_xyz[:, 2] = (
            point_world[:, 0] * r_end[0, 2] + point_world[:, 1] * r_end[1, 2] + point_world[:, 2] * r_end[2, 2]
        )
        finite = cloud["finite_time"]
        corrected_xyz[~finite, 0] = x[~finite]
        corrected_xyz[~finite, 1] = y[~finite]
        corrected_xyz[~finite, 2] = z[~finite]
        xyz_transform_ms = (time.perf_counter() - xyz_start) * 1000.0
        encode_start = time.perf_counter()
        selected_indices = cloud["indices"]
        output = message.__class__()
        output.header = message.header
        output.fields = message.fields
        output.is_bigendian = message.is_bigendian
        output.point_step = message.point_step
        output.is_dense = message.is_dense
        if selected_indices is None:
            output.height = message.height
            output.width = message.width
            output.row_step = message.row_step
            output_data = bytearray(message.data)
            output_message = message
        else:
            raw_records = np.frombuffer(message.data, dtype=np.uint8).reshape(-1, message.point_step)
            output_data = bytearray(np.ascontiguousarray(raw_records[selected_indices]).tobytes())
            output.height = 1
            output.width = int(selected_indices.size)
            output.row_step = output.width * output.point_step
            output_message = output
        output_views = {
            field.name: self._field_view(output_data, output_message, field)
            for field in output.fields
            if field.name in {"x", "y", "z"}
        }
        output_views["x"][:] = corrected_xyz[:, 0].reshape(output.height, output.width)
        output_views["y"][:] = corrected_xyz[:, 1].reshape(output.height, output.width)
        output_views["z"][:] = corrected_xyz[:, 2].reshape(output.height, output.width)
        output.data = bytes(output_data)
        encode_ms = (time.perf_counter() - encode_start) * 1000.0
        publish_start = time.perf_counter()
        self.cloud_pub.publish(output)
        publish_end = time.perf_counter()
        publish_ms = (publish_end - publish_start) * 1000.0
        with self.diag_lock:
            drop_count = self.diag_drop_count
        self.get_logger().info(
            "[DESKEW_FRAME] "
            f"frame_id={work.frame_id} "
            f"cloud_stamp={stamp_seconds(message.header.stamp):.9f} "
            f"queue_before={work.queue_before} "
            f"queue_after={work.queue_after} "
            f"imu_buffer_size={imu_buffer_size} "
            f"imu_count_used={len(samples)} "
            f"point_count={times.size} "
            f"decode_ms={decode_ms:.3f} "
            f"imu_snapshot_ms={snapshot_ms:.3f} "
            f"deskew_math_ms={deskew_math_ms:.3f} "
            f"xyz_transform_ms={xyz_transform_ms:.3f} "
            f"encode_ms={encode_ms:.3f} "
            f"wait_imu_ms={wait_imu_ms:.3f} "
            f"process_ms={(publish_start - process_start) * 1000.0:.3f} "
            f"publish_ms={publish_ms:.3f} "
            f"total_latency_ms={(publish_end - work.received_at) * 1000.0:.3f} "
            f"drop_count={drop_count}"
        )
        process_time = time.perf_counter() - process_start
        with self.diag_lock:
            self.diag_processed_count += 1
            self.diag_process_time_sum += process_time
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_diag >= float(self.get_parameter("diagnostic_period_seconds").value):
            rotation_angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r_end) - 1.0) * 0.5))))
            translation = float(np.linalg.norm(p_end))
            self.get_logger().info(
                "DESKEW_DIAG "
                f"scan_duration={float(np.max(times) - np.min(times)):.6f} "
                f"imu_count={len(samples)} "
                f"rotation_compensation_deg={rotation_angle:.6f} "
                f"translation_compensation={translation:.6f} "
                f"process_time_ms={(time.perf_counter() - process_start) * 1000.0:.3f} "
                f"cloud_queue_size={self.cloud_queue.qsize()} "
                f"imu_buffer_size={len(self.imu_buffer)} "
                f"imu_count_used={len(samples)}"
            )
            self.last_diag = now

    def publish_status(self) -> None:
        with self.diag_lock:
            now = time.monotonic()
            if not hasattr(self, "status_last_time"):
                self.status_last_time = now
                self.status_last_imu = self.diag_imu_count
                self.status_last_cloud = self.diag_cloud_count
                self.status_last_processed = self.diag_processed_count
                self.status_last_process_sum = self.diag_process_time_sum
                return
            elapsed = max(now - self.status_last_time, 1e-6)
            imu_rate = (self.diag_imu_count - self.status_last_imu) / elapsed
            cloud_rate = (self.diag_cloud_count - self.status_last_cloud) / elapsed
            processed = self.diag_processed_count - self.status_last_processed
            process_sum = self.diag_process_time_sum - self.status_last_process_sum
            avg_process_ms = process_sum * 1000.0 / processed if processed else 0.0
            self.status_last_time = now
            self.status_last_imu = self.diag_imu_count
            self.status_last_cloud = self.diag_cloud_count
            self.status_last_processed = self.diag_processed_count
            self.status_last_process_sum = self.diag_process_time_sum
        with self.buffer_lock:
            buffer_size = len(self.imu_buffer)
        self.get_logger().info(
            "DESKEW_STATUS enabled=true "
            f"imu_rate={imu_rate:.3f} cloud_rate={cloud_rate:.3f} "
            f"avg_process_time_ms={avg_process_ms:.3f} "
            f"buffer_size={buffer_size}"
        )

    def close(self) -> None:
        self.worker_stop.set()
        with self.buffer_lock:
            self.buffer_lock.notify_all()
        self.worker.join(timeout=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarDeskewNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

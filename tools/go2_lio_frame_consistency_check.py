#!/usr/bin/env python3
"""Read-only LiDAR/IMU body-frame consistency check.

The tool estimates a dominant plane from each cloud without assuming that
the cloud's z axis is vertical.  It never publishes and does not touch any
robot-control API.
"""

import argparse
import math
import statistics
import time


RAW_CLOUD_TOPIC = "/utlidar/cloud"
RAW_IMU_TOPIC = "/utlidar/imu"
TRANSFORMED_CLOUD_TOPIC = "/sea_nav/lio/transformed_cloud"
TRANSFORMED_IMU_TOPIC = "/sea_nav/lio/transformed_raw_imu"


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def fmt(value):
    return "NA" if value is None else f"{float(value):.9f}"


def fmt_vector(vector):
    if vector is None:
        return "NA"
    return "[" + ",".join(f"{float(value):.9f}" for value in vector) + "]"


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def normalize(vector, numpy):
    norm = float(numpy.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None, None
    return vector / norm, norm


def plane_from_three(first, second, third, numpy):
    normal = numpy.cross(second - first, third - first)
    normal, norm = normalize(normal, numpy)
    if normal is None:
        return None
    return normal, -float(numpy.dot(normal, first))


def fit_plane(points, numpy, rng, threshold, iterations):
    """Return the dominant RANSAC plane for one frame, or None."""
    if len(points) < 6:
        return None

    best = None
    count = len(points)
    for _ in range(iterations):
        indices = rng.choice(count, size=3, replace=False)
        model = plane_from_three(points[indices[0]], points[indices[1]], points[indices[2]], numpy)
        if model is None:
            continue
        normal, offset = model
        distances = numpy.abs(points @ normal + offset)
        inliers = distances <= threshold
        inlier_count = int(numpy.count_nonzero(inliers))
        if inlier_count < 6:
            continue
        inlier_points = points[inliers]
        centered = inlier_points - inlier_points.mean(axis=0)
        eigenvalues = numpy.linalg.eigvalsh(centered.T @ centered)
        extent_score = float(eigenvalues[-1] * eigenvalues[-2])
        score = (inlier_count, extent_score)
        if best is None or score > best["score"]:
            best = {
                "normal": normal,
                "offset": offset,
                "inliers": inlier_points,
                "total": count,
                "score": score,
            }

    if best is None:
        return None

    centered = best["inliers"] - best["inliers"].mean(axis=0)
    _, _, vh = numpy.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal, _ = normalize(normal, numpy)
    offset = -float(numpy.dot(normal, best["inliers"].mean(axis=0)))
    residuals = numpy.abs(best["inliers"] @ normal + offset)
    best["normal"] = normal
    best["offset"] = offset
    best["rms"] = float(numpy.sqrt(numpy.mean(residuals * residuals)))
    return best


class PlaneAccumulator:
    def __init__(self, numpy, max_points, max_range, threshold, iterations):
        self.numpy = numpy
        self.max_points = max_points
        self.max_range = max_range
        self.threshold = threshold
        self.iterations = iterations
        self.rng = numpy.random.default_rng(20260822)
        self.frames = 0
        self.total_points = 0
        self.fits = []
        self.frame_ids = []

    def add_message(self, message, point_reader):
        points = point_reader(message, self.numpy, self.max_points, self.max_range)
        if points is None or len(points) < 6:
            return
        self.frames += 1
        self.total_points += len(points)
        frame_id = str(message.header.frame_id)
        if frame_id and frame_id not in self.frame_ids:
            self.frame_ids.append(frame_id)
        fit = fit_plane(points, self.numpy, self.rng, self.threshold, self.iterations)
        if fit is not None:
            self.fits.append(fit)

    def aggregate(self):
        if not self.fits:
            return None

        best = max(self.fits, key=lambda item: item["score"])
        reference = best["normal"]
        normals = []
        offsets = []
        selected = []
        for fit in self.fits:
            normal = fit["normal"].copy()
            offset = fit["offset"]
            if float(self.numpy.dot(normal, reference)) < 0.0:
                normal = -normal
                offset = -offset
            if float(self.numpy.dot(normal, reference)) >= math.cos(math.radians(25.0)):
                normals.append(normal)
                offsets.append(offset)
                selected.append(fit)

        normal, _ = normalize(self.numpy.median(self.numpy.asarray(normals), axis=0), self.numpy)
        offset = float(self.numpy.median(self.numpy.asarray(offsets)))
        inlier_points = self.numpy.concatenate([fit["inliers"] for fit in selected], axis=0)
        residuals = self.numpy.abs(inlier_points @ normal + offset)
        return {
            "normal": normal,
            "offset": offset,
            "inlier_count": int(len(inlier_points)),
            "inlier_ratio": float(len(inlier_points) / max(1, sum(fit["total"] for fit in selected))),
            "rms": float(self.numpy.sqrt(self.numpy.mean(residuals * residuals))),
            "frames": len(selected),
        }


def point_reader(message, numpy, max_points, max_range):
    from sensor_msgs_py import point_cloud2 as pc2

    try:
        rows = pc2.read_points_list(
            message,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
    except Exception:
        return None

    points = []
    for row in rows:
        try:
            values = (row[0], row[1], row[2])
        except (IndexError, TypeError):
            values = (row.x, row.y, row.z)
        try:
            point = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in point):
            range_3d = math.sqrt(sum(value * value for value in point))
            if 0.10 <= range_3d <= max_range:
                points.append(point)

    if len(points) > max_points:
        indices = numpy.linspace(0, len(points) - 1, max_points, dtype=int)
        points = [points[index] for index in indices]
    return numpy.asarray(points, dtype=float)


class FrameCheck:
    def __init__(self, args, numpy):
        self.args = args
        self.numpy = numpy
        self.transformed_cloud = PlaneAccumulator(
            numpy, args.max_points, args.max_range, args.ransac_threshold, args.ransac_iterations
        )
        self.raw_cloud = PlaneAccumulator(
            numpy, args.max_points, args.max_range, args.ransac_threshold, args.ransac_iterations
        )
        self.transformed_acc = []
        self.raw_acc = []
        self.transformed_imu_count = 0
        self.raw_imu_count = 0
        self.node = None
        self.rclpy = None

    def transformed_cloud_callback(self, message):
        self.transformed_cloud.add_message(message, point_reader)

    def raw_cloud_callback(self, message):
        self.raw_cloud.add_message(message, point_reader)

    def transformed_imu_callback(self, message):
        self.transformed_imu_count += 1
        self.transformed_acc.append([
            float(message.linear_acceleration.x),
            float(message.linear_acceleration.y),
            float(message.linear_acceleration.z),
        ])

    def raw_imu_callback(self, message):
        self.raw_imu_count += 1
        self.raw_acc.append([
            float(message.linear_acceleration.x),
            float(message.linear_acceleration.y),
            float(message.linear_acceleration.z),
        ])

    def setup(self):
        import rclpy
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Imu, PointCloud2

        self.rclpy = rclpy
        self.node = rclpy.create_node("go2_lio_frame_consistency_check")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.node.create_subscription(
            PointCloud2, TRANSFORMED_CLOUD_TOPIC, self.transformed_cloud_callback, qos
        )
        self.node.create_subscription(
            Imu, TRANSFORMED_IMU_TOPIC, self.transformed_imu_callback, qos
        )
        self.node.create_subscription(PointCloud2, RAW_CLOUD_TOPIC, self.raw_cloud_callback, qos)
        self.node.create_subscription(Imu, RAW_IMU_TOPIC, self.raw_imu_callback, qos)

    def run(self):
        started = time.monotonic()
        while self.rclpy.ok() and time.monotonic() - started < self.args.duration:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)

    def report_direction(self, label, samples):
        if not samples:
            print(f"{label}_ACC_MEAN=NA")
            print(f"{label}_ACC_NORM=NA")
            print(f"{label}_UP_VECTOR=NA")
            return None
        values = self.numpy.asarray(samples, dtype=float)
        mean = values.mean(axis=0)
        direction, norm = normalize(mean, self.numpy)
        print(f"{label}_ACC_MEAN={fmt_vector(mean)}")
        print(f"{label}_ACC_NORM={fmt(norm)}")
        print(f"{label}_UP_VECTOR={fmt_vector(direction)}")
        return direction

    def report(self):
        transformed_up = self.report_direction("IMU", self.transformed_acc)
        self.report_direction("RAW_IMU", self.raw_acc)

        aggregate = self.transformed_cloud.aggregate()
        if aggregate is None:
            print("CLOUD_PLANE_NORMAL=NA")
            print("PLANE_INLIER_COUNT=0")
            print("PLANE_INLIER_RATIO=NA")
            print("PLANE_FIT_RMS=NA")
            print("ANGLE_MEAN_DEG=NA")
            print("ANGLE_P50_DEG=NA")
            print("ANGLE_P95_DEG=NA")
            print("ANGLE_MAX_DEG=NA")
        else:
            normal = aggregate["normal"].copy()
            if transformed_up is not None and float(self.numpy.dot(normal, transformed_up)) < 0.0:
                normal = -normal
            print(f"CLOUD_PLANE_NORMAL={fmt_vector(normal)}")
            print(f"PLANE_INLIER_COUNT={aggregate['inlier_count']}")
            print(f"PLANE_INLIER_RATIO={aggregate['inlier_ratio']:.9f}")
            print(f"PLANE_FIT_RMS={aggregate['rms']:.9f}")

            angles = []
            if transformed_up is not None:
                for fit in self.transformed_cloud.fits:
                    candidate = fit["normal"]
                    cosine = abs(float(self.numpy.dot(candidate, transformed_up)))
                    cosine = max(-1.0, min(1.0, cosine))
                    angles.append(math.degrees(math.acos(cosine)))
            print(f"ANGLE_MEAN_DEG={fmt(statistics.mean(angles) if angles else None)}")
            print(f"ANGLE_P50_DEG={fmt(percentile(angles, 0.50))}")
            print(f"ANGLE_P95_DEG={fmt(percentile(angles, 0.95))}")
            print(f"ANGLE_MAX_DEG={fmt(max(angles) if angles else None)}")

        raw_aggregate = self.raw_cloud.aggregate()
        print(f"RAW_CLOUD_PLANE_NORMAL={fmt_vector(raw_aggregate['normal']) if raw_aggregate else 'NA'}")
        print(f"TRANSFORMED_CLOUD_FRAMES={self.transformed_cloud.frames}")
        print(f"TRANSFORMED_CLOUD_PLANE_FRAMES={len(self.transformed_cloud.fits)}")
        print(f"TRANSFORMED_CLOUD_FRAME_IDS={','.join(self.transformed_cloud.frame_ids) or 'NA'}")
        print(f"RAW_CLOUD_FRAMES={self.raw_cloud.frames}")
        print(f"RAW_CLOUD_PLANE_FRAMES={len(self.raw_cloud.fits)}")
        print(f"RAW_CLOUD_FRAME_IDS={','.join(self.raw_cloud.frame_ids) or 'NA'}")
        print(f"RAW_IMU_SAMPLES={self.raw_imu_count}")
        print(f"TRANSFORMED_IMU_SAMPLES={self.transformed_imu_count}")

        p95 = percentile(angles, 0.95) if aggregate is not None and transformed_up is not None else None
        if p95 is None:
            status = "UNKNOWN"
        elif p95 < 5.0:
            status = "CONSISTENT"
        elif p95 <= 15.0:
            status = "SUSPICIOUS"
        else:
            status = "MISMATCH"
        print(f"FRAME_ROTATION_STATUS={status}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--max-points", type=int, default=600)
    parser.add_argument("--max-range", type=float, default=6.0)
    parser.add_argument("--ransac-threshold", type=float, default=0.04)
    parser.add_argument("--ransac-iterations", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy
    import rclpy

    rclpy.init(args=None)
    check = FrameCheck(args, numpy)
    check.setup()
    try:
        check.run()
        check.report()
    finally:
        check.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

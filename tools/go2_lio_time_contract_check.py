#!/usr/bin/env python3
"""Read-only Go2 LiDAR/IMU timestamp and PointCloud2 time contract check."""

import argparse
import json
import math
import signal
import statistics
import struct
import time


RAW_CLOUD_TOPIC = "/utlidar/cloud"
RAW_IMU_TOPIC = "/utlidar/imu"
TRANSFORMED_CLOUD_TOPIC = "/sea_nav/lio/transformed_cloud"
TRANSFORMED_IMU_TOPIC = "/sea_nav/lio/transformed_raw_imu"


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


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


def format_value(value):
    return "NA" if value is None else f"{value:.9f}"


def nearest_absolute_offsets(left, right):
    """Return absolute nearest-neighbor time differences in seconds."""
    if not left or not right:
        return []
    right_sorted = sorted(right)
    result = []
    cursor = 0
    for value in left:
        while cursor + 1 < len(right_sorted) and abs(right_sorted[cursor + 1] - value) <= abs(right_sorted[cursor] - value):
            cursor += 1
        result.append(abs(right_sorted[cursor] - value))
    return result


def monotonic(values):
    if len(values) < 2:
        return None
    return all(current >= previous for previous, current in zip(values, values[1:]))


def infer_time_unit(point_span_values, cloud_period_seconds):
    """Infer the field unit by comparing converted scan spans to scan period."""
    if not point_span_values or not cloud_period_seconds or cloud_period_seconds <= 0.0:
        return "AMBIGUOUS", None, None

    median_span = statistics.median(point_span_values)
    if not math.isfinite(median_span) or median_span <= 0.0:
        return "AMBIGUOUS", None, None

    candidates = (
        ("SECONDS", 1.0),
        ("MILLISECONDS", 1e-3),
        ("MICROSECONDS", 1e-6),
        ("NANOSECONDS", 1e-9),
    )
    scored = []
    for name, scale in candidates:
        ratio = median_span * scale / cloud_period_seconds
        if ratio > 0.0 and math.isfinite(ratio):
            scored.append((abs(math.log(ratio)), name, scale, ratio))
    if not scored:
        return "AMBIGUOUS", None, None
    _, name, scale, ratio = min(scored)
    if ratio < 0.25 or ratio > 4.0:
        return "AMBIGUOUS", None, ratio
    return name, scale, ratio


_STRUCT_FORMATS = {
    1: "b",   # INT8
    2: "B",   # UINT8
    3: "h",   # INT16
    4: "H",   # UINT16
    5: "i",   # INT32
    6: "I",   # UINT32
    7: "f",   # FLOAT32
    8: "d",   # FLOAT64
}


def sampled_point_field_values(message, field_name, max_samples=128):
    """Read a scalar PointCloud2 field without converting or publishing the cloud."""
    field = next((item for item in message.fields if item.name == field_name), None)
    if field is None:
        return []
    code = _STRUCT_FORMATS.get(int(field.datatype))
    if code is None or int(field.count) < 1:
        return []

    width = int(message.width)
    height = int(message.height)
    total = width * height
    if total <= 0 or int(message.point_step) <= 0:
        return []
    samples = min(total, max(1, int(max_samples)))
    indices = [int(index * (total - 1) / max(1, samples - 1)) for index in range(samples)]
    prefix = ">" if bool(message.is_bigendian) else "<"
    format_string = prefix + code
    value_size = struct.calcsize(format_string)
    point_step = int(message.point_step)
    row_step = int(message.row_step) or point_step * width
    offset = int(field.offset)
    data = message.data
    values = []
    for index in indices:
        row, column = divmod(index, width)
        byte_offset = row * row_step + column * point_step + offset
        if byte_offset < 0 or byte_offset + value_size > len(data):
            continue
        values.append(float(struct.unpack_from(format_string, data, byte_offset)[0]))
    return values


class TimeContract:
    def __init__(self, args):
        self.args = args
        self.cloud_stamps = []
        self.transformed_cloud_stamps = []
        self.raw_imu_stamps = []
        self.transformed_imu_stamps = []
        self.cloud_periods = []
        self.point_spans = []
        self.point_values = []
        self.point_sequences = []
        self.point_negative = False
        self.point_nonfinite = False
        self.fields = None
        self.point_field = None
        self.errors = []
        self.node = None

    def cloud_callback(self, message):
        stamp = stamp_seconds(message.header.stamp)
        if self.cloud_stamps:
            self.cloud_periods.append(stamp - self.cloud_stamps[-1])
        self.cloud_stamps.append(stamp)
        if self.fields is None:
            self.fields = [
                {
                    "name": str(field.name),
                    "datatype": int(field.datatype),
                    "offset": int(field.offset),
                    "count": int(field.count),
                }
                for field in message.fields
            ]
            names = {field["name"] for field in self.fields}
            for candidate in ("time", "t", "timestamp"):
                if candidate in names:
                    self.point_field = candidate
                    break
            print("POINTCLOUD_FIELDS=" + json.dumps(self.fields, separators=(",", ":")), flush=True)
            print("POINT_TIME_FIELD=" + (self.point_field or "ABSENT"), flush=True)

        if not self.point_field:
            return
        values = sampled_point_field_values(message, self.point_field, self.args.max_points)
        finite = [value for value in values if math.isfinite(value)]
        if len(finite) != len(values):
            self.point_nonfinite = True
        if any(value < 0.0 for value in finite):
            self.point_negative = True
        if finite:
            self.point_values.extend(finite)
            self.point_spans.append(max(finite) - min(finite))
            self.point_sequences.append(finite)

    def raw_imu_callback(self, message):
        self.raw_imu_stamps.append(stamp_seconds(message.header.stamp))

    def transformed_cloud_callback(self, message):
        self.transformed_cloud_stamps.append(stamp_seconds(message.header.stamp))

    def transformed_imu_callback(self, message):
        self.transformed_imu_stamps.append(stamp_seconds(message.header.stamp))

    def setup(self):
        import rclpy
        from sensor_msgs.msg import Imu, PointCloud2
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("go2_lio_time_contract_check")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.node.create_subscription(PointCloud2, RAW_CLOUD_TOPIC, self.cloud_callback, qos)
        self.node.create_subscription(PointCloud2, TRANSFORMED_CLOUD_TOPIC, self.transformed_cloud_callback, qos)
        self.node.create_subscription(Imu, RAW_IMU_TOPIC, self.raw_imu_callback, qos)
        self.node.create_subscription(Imu, TRANSFORMED_IMU_TOPIC, self.transformed_imu_callback, qos)

    def run(self):
        import rclpy

        started = time.monotonic()
        while rclpy.ok() and time.monotonic() - started < self.args.duration:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def close(self):
        import rclpy

        if self.node is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def report(self):
        if self.fields is None:
            print("POINTCLOUD_FIELDS=NOT_AVAILABLE")
            print("POINT_TIME_FIELD=ABSENT")
        cloud_periods = [value for value in self.cloud_periods if value > 0.0 and math.isfinite(value)]
        cloud_period = statistics.mean(cloud_periods) if cloud_periods else None
        cloud_rate = (len(self.cloud_stamps) - 1) / (self.cloud_stamps[-1] - self.cloud_stamps[0]) if len(self.cloud_stamps) >= 2 and self.cloud_stamps[-1] > self.cloud_stamps[0] else 0.0
        transformed_cloud_rate = stream_rate(self.transformed_cloud_stamps)
        raw_rate = (len(self.raw_imu_stamps) - 1) / (self.raw_imu_stamps[-1] - self.raw_imu_stamps[0]) if len(self.raw_imu_stamps) >= 2 and self.raw_imu_stamps[-1] > self.raw_imu_stamps[0] else 0.0
        transformed_rate = (len(self.transformed_imu_stamps) - 1) / (self.transformed_imu_stamps[-1] - self.transformed_imu_stamps[0]) if len(self.transformed_imu_stamps) >= 2 and self.transformed_imu_stamps[-1] > self.transformed_imu_stamps[0] else 0.0

        raw_offsets = nearest_absolute_offsets(self.cloud_stamps, self.raw_imu_stamps)
        transformed_offsets = nearest_absolute_offsets(self.transformed_cloud_stamps, self.transformed_imu_stamps)
        raw_to_transformed_cloud = ordered_signed_offsets(self.cloud_stamps, self.transformed_cloud_stamps)
        raw_to_transformed_imu = ordered_signed_offsets(self.raw_imu_stamps, self.transformed_imu_stamps)
        unit, scale, ratio = infer_time_unit(self.point_spans, cloud_period)
        converted_spans = [value * scale for value in self.point_spans] if scale else []
        converted_mean = statistics.mean(converted_spans) if converted_spans else None
        converted_p95 = percentile(converted_spans, 0.95)
        ratio = converted_mean / cloud_period if converted_mean is not None and cloud_period else ratio

        point_order = None
        if self.point_sequences:
            ordered = sum(1 for values in self.point_sequences if monotonic(values))
            point_order = ordered >= max(1, math.ceil(0.8 * len(self.point_sequences)))
        reset = None
        if self.point_spans:
            reset = sum(
                1 for values, span in zip(self.point_sequences, self.point_spans)
                if span > 0.0 and min(values) <= max(1e-9, span * 0.1)
            ) >= max(1, math.ceil(0.8 * len(self.point_spans)))

        time_contract = "AMBIGUOUS"
        if unit == "SECONDS":
            if ratio is not None and 0.25 <= ratio <= 4.0 and point_order is not False and not self.point_negative and not self.point_nonfinite:
                time_contract = "PASS"
            else:
                time_contract = "FAIL"
        elif unit != "AMBIGUOUS":
            time_contract = "FAIL"
        deskew_suspect = time_contract == "FAIL"
        sync_suspect = bool(
            cloud_period
            and (transformed_offsets or raw_offsets)
            and percentile(transformed_offsets or raw_offsets, 0.95) > max(0.010, cloud_period * 0.5)
        )

        transformed_cloud_check = check_nearest_offset(transformed_offsets, threshold_s=0.010)
        raw_cloud_check = check_offset(raw_to_transformed_cloud, threshold_s=0.005)
        raw_imu_check = check_offset(raw_to_transformed_imu, threshold_s=0.005)
        cloud_offset_mean = mean_value(raw_to_transformed_cloud)
        imu_offset_mean = mean_value(raw_to_transformed_imu)
        offset_diff_ms = (
            (cloud_offset_mean - imu_offset_mean) * 1000.0
            if cloud_offset_mean is not None and imu_offset_mean is not None
            else None
        )
        offset_diff_check = (
            "PASS"
            if offset_diff_ms is not None
            and abs(offset_diff_ms) <= 5.0
            and check_offset_std(raw_to_transformed_cloud, threshold_s=0.005)
            and check_offset_std(raw_to_transformed_imu, threshold_s=0.005)
            else "FAIL" if offset_diff_ms is not None else "NOT_AVAILABLE"
        )
        transform_contract = combined_check(transformed_cloud_check, offset_diff_check)
        time_sync_root_cause = (
            "NO" if transform_contract == "PASS"
            else "YES" if transform_contract == "FAIL"
            else "UNKNOWN"
        )

        print("CLOUD_RATE_HZ=" + f"{cloud_rate:.6f}")
        print("TRANSFORMED_CLOUD_RATE_HZ=" + f"{transformed_cloud_rate:.6f}")
        print("CLOUD_HEADER_MONOTONIC=" + format_bool(monotonic(self.cloud_stamps)))
        print("CLOUD_PERIOD_MEAN_MS=" + format_value(statistics.mean(cloud_periods) * 1000.0 if cloud_periods else None))
        print("CLOUD_PERIOD_P95_MS=" + format_value(percentile([value * 1000.0 for value in cloud_periods], 0.95)))
        print("RAW_IMU_RATE_HZ=" + f"{raw_rate:.6f}")
        print("TRANSFORMED_IMU_RATE_HZ=" + f"{transformed_rate:.6f}")
        print("RAW_IMU_HEADER_MONOTONIC=" + format_bool(monotonic(self.raw_imu_stamps)))
        print("TRANSFORMED_IMU_HEADER_MONOTONIC=" + format_bool(monotonic(self.transformed_imu_stamps)))
        print_stats("RAW_LIDAR_IMU_OFFSET", raw_offsets, scale=1000.0)
        print_stats("TRANSFORMED_LIDAR_IMU_OFFSET", transformed_offsets, scale=1000.0)
        print_stats("TRANSFORM_HEADER_OFFSET", raw_to_transformed_imu, scale=1000.0, include_std=True)
        print_stats("TRANSFORMED_CLOUD_IMU_OFFSET", transformed_offsets, scale=1000.0)
        print_mean_std("RAW_TO_TRANSFORMED_CLOUD_OFFSET", raw_to_transformed_cloud)
        print_mean_std("RAW_TO_TRANSFORMED_IMU_OFFSET", raw_to_transformed_imu)
        print("CLOUD_IMU_TRANSFORM_OFFSET_DIFF_MS=" + format_value(offset_diff_ms))
        print("TRANSFORMED_CLOUD_IMU_CHECK=" + transformed_cloud_check)
        print("RAW_TO_TRANSFORMED_CLOUD_CHECK=" + raw_cloud_check)
        print("RAW_TO_TRANSFORMED_IMU_CHECK=" + raw_imu_check)
        print("OFFSET_DIFF_CHECK=" + offset_diff_check)
        print("TRANSFORM_TIME_SYNC_CONTRACT=" + transform_contract)
        print("TIME_SYNC_ROOT_CAUSE=" + time_sync_root_cause)
        print("POINT_TIME_MIN=" + format_value(min(self.point_values) if self.point_values else None))
        print("POINT_TIME_MAX=" + format_value(max(self.point_values) if self.point_values else None))
        print("POINT_TIME_SPAN_MEAN=" + format_value(statistics.mean(self.point_spans) if self.point_spans else None))
        print("POINT_TIME_SPAN_P95=" + format_value(percentile(self.point_spans, 0.95)))
        print("POINT_TIME_SPAN_MAX=" + format_value(max(self.point_spans) if self.point_spans else None))
        for index, value in enumerate(self.point_values[:10], 1):
            print(f"POINT_TIME_SAMPLE_{index}=" + format_value(value))
        print("POINT_TIME_UNIT_INFERRED=" + unit)
        median_span = statistics.median(self.point_spans) if self.point_spans else None
        print(
            "POINT_TIME_UNIT_EVIDENCE="
            + json.dumps(
                {
                    "median_raw_span": median_span,
                    "cloud_period_s": cloud_period,
                    "selected_converted_ratio": ratio,
                    "point_lio_timestamp_unit_0_means": "SECONDS",
                },
                separators=(",", ":"),
            )
        )
        print("POINT_TIME_SPAN_TO_CLOUD_PERIOD_RATIO=" + format_value(ratio))
        print("POINT_TIME_NEGATIVE=" + str(self.point_negative))
        print("POINT_TIME_NONFINITE=" + str(self.point_nonfinite))
        print("POINT_TIME_RESET_PER_FRAME=" + format_bool(reset))
        print("POINT_TIME_ORDER=" + format_bool(point_order))
        print("DESKEW_TIME_CONTRACT=" + time_contract)
        print("TIME_SYNC_SUSPECT=" + ("YES" if sync_suspect else "NO"))
        print("DESKEW_SUSPECT=" + ("YES" if deskew_suspect else "NO"))


def stream_rate(values):
    if len(values) < 2 or values[-1] <= values[0]:
        return 0.0
    return (len(values) - 1) / (values[-1] - values[0])


def ordered_signed_offsets(source, target):
    """Pair two order-preserving streams and return target_stamp - source_stamp."""
    if not source or not target:
        return []
    count = min(len(source), len(target))
    if count == 1:
        source_indices = [0]
        target_indices = [0]
    else:
        source_indices = [round(index * (len(source) - 1) / (count - 1)) for index in range(count)]
        target_indices = [round(index * (len(target) - 1) / (count - 1)) for index in range(count)]
    return [target[target_index] - source[source_index] for source_index, target_index in zip(source_indices, target_indices)]


def mean_value(values):
    return statistics.mean(values) if values else None


def std_value(values):
    return statistics.pstdev(values) if len(values) >= 2 else (0.0 if values else None)


def check_offset(values, threshold_s):
    if not values:
        return "NOT_AVAILABLE"
    return "PASS" if check_offset_std(values, threshold_s) else "FAIL"


def check_nearest_offset(values, threshold_s):
    if not values:
        return "NOT_AVAILABLE"
    p95 = percentile(values, 0.95)
    return "PASS" if p95 is not None and p95 <= threshold_s else "FAIL"


def check_offset_std(values, threshold_s):
    std = std_value(values)
    return std is not None and std <= threshold_s


def combined_check(first, second):
    if first == "FAIL" or second == "FAIL":
        return "FAIL"
    if first == "PASS" and second == "PASS":
        return "PASS"
    return "AMBIGUOUS"


def format_bool(value):
    if value is None:
        return "UNKNOWN"
    return "YES" if value else "NO"


def print_stats(prefix, values, scale=1.0, include_std=False):
    scaled = [value * scale for value in values]
    print(f"{prefix}_MEAN_MS=" + format_value(statistics.mean(scaled) if scaled else None))
    print(f"{prefix}_P50_MS=" + format_value(percentile(scaled, 0.50)))
    print(f"{prefix}_P95_MS=" + format_value(percentile(scaled, 0.95)))
    print(f"{prefix}_MAX_MS=" + format_value(max(scaled) if scaled else None))
    if include_std:
        std = statistics.pstdev(scaled) if len(scaled) >= 2 else (0.0 if scaled else None)
        print(f"{prefix}_STD_MS=" + format_value(std))


def print_mean_std(prefix, values):
    scaled = [value * 1000.0 for value in values]
    print(f"{prefix}_MEAN_MS=" + format_value(mean_value(scaled)))
    print(f"{prefix}_STD_MS=" + format_value(std_value(scaled)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--max-points", type=int, default=128)
    args = parser.parse_args(argv)
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.max_points <= 0:
        parser.error("--max-points must be positive")

    checker = TimeContract(args)
    checker.setup()

    def stop(_signum, _frame):
        import rclpy
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        checker.run()
    finally:
        checker.report()
        checker.close()


if __name__ == "__main__":
    main()

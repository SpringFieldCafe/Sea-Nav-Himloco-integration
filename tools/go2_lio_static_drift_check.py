#!/usr/bin/env python3
"""Read-only 10-second Point-LIO static drift and transformed IMU recorder."""

import argparse
from collections import deque
import math
import signal
import statistics
import time


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def angle_delta(current, previous):
    return math.atan2(math.sin(current - previous), math.cos(current - previous))


def mean_std(values):
    if not values:
        return None, None
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


class StaticRecorder:
    def __init__(self, args):
        import rclpy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        self.rclpy = rclpy
        self.args = args
        self.node = rclpy.create_node('go2_lio_static_drift_check')
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.odom = []
        self.imu = []
        self.odom_window = deque()
        self.max_1s_position_change = 0.0
        self.max_1s_yaw_change = 0.0
        self.node.create_subscription(Odometry, args.odom_topic, self.odom_callback, qos)
        self.node.create_subscription(Imu, args.imu_topic, self.imu_callback, qos)

    def odom_callback(self, message):
        stamp = stamp_seconds(message.header.stamp)
        if stamp <= 0.0:
            stamp = time.monotonic()
        pose = message.pose.pose
        sample = (
            stamp,
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            yaw_from_quaternion(pose.orientation),
        )
        self.odom.append(sample)
        self.odom_window.append(sample)
        while self.odom_window and stamp - self.odom_window[0][0] > 1.0:
            self.odom_window.popleft()
        if self.odom_window:
            old = self.odom_window[0]
            self.max_1s_position_change = max(
                self.max_1s_position_change,
                math.sqrt(sum((sample[index] - old[index]) ** 2 for index in (1, 2, 3))),
            )
            self.max_1s_yaw_change = max(
                self.max_1s_yaw_change,
                abs(angle_delta(sample[4], old[4])),
            )

    def imu_callback(self, message):
        self.imu.append((
            float(message.angular_velocity.x),
            float(message.angular_velocity.y),
            float(message.angular_velocity.z),
            float(message.linear_acceleration.x),
            float(message.linear_acceleration.y),
            float(message.linear_acceleration.z),
        ))

    def run(self):
        started = time.monotonic()
        while self.rclpy.ok() and time.monotonic() - started < self.args.duration:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)

    def report(self):
        if self.odom:
            start = self.odom[0]
            end = self.odom[-1]
            position_drift = math.sqrt(sum((end[index] - start[index]) ** 2 for index in (1, 2, 3)))
            yaw_drift = abs(angle_delta(end[4], start[4]))
            rate = (len(self.odom) - 1) / (end[0] - start[0]) if end[0] > start[0] else 0.0
            print(f"START_X={start[1]:.9f}")
            print(f"START_Y={start[2]:.9f}")
            print(f"START_YAW_DEG={math.degrees(start[4]):.9f}")
            print(f"END_X={end[1]:.9f}")
            print(f"END_Y={end[2]:.9f}")
            print(f"END_YAW_DEG={math.degrees(end[4]):.9f}")
            print(f"POSITION_DRIFT_CM={position_drift * 100.0:.9f}")
            print(f"YAW_DRIFT_DEG={math.degrees(yaw_drift):.9f}")
            print(f"MAX_1S_POSITION_CHANGE_CM={self.max_1s_position_change * 100.0:.9f}")
            print(f"MAX_1S_YAW_CHANGE_DEG={math.degrees(self.max_1s_yaw_change):.9f}")
            print(f"ODOM_RATE_HZ={rate:.6f}")
        else:
            for key in (
                'START_X', 'START_Y', 'START_YAW_DEG', 'END_X', 'END_Y',
                'END_YAW_DEG', 'POSITION_DRIFT_CM', 'YAW_DRIFT_DEG',
                'MAX_1S_POSITION_CHANGE_CM', 'MAX_1S_YAW_CHANGE_DEG', 'ODOM_RATE_HZ',
            ):
                print(f"{key}=NA")

        names = ('GYRO_MEAN_X', 'GYRO_MEAN_Y', 'GYRO_MEAN_Z', 'ACCEL_MEAN_X', 'ACCEL_MEAN_Y', 'ACCEL_MEAN_Z')
        std_names = ('GYRO_STD_X', 'GYRO_STD_Y', 'GYRO_STD_Z', 'ACCEL_STD_X', 'ACCEL_STD_Y', 'ACCEL_STD_Z')
        columns = list(zip(*self.imu)) if self.imu else []
        for name, values in zip(names, columns):
            mean, _ = mean_std(values)
            print(f"{name}={mean:.9f}" if mean is not None else f"{name}=NA")
        for name, values in zip(std_names, columns):
            _, std = mean_std(values)
            print(f"{name}={std:.9f}" if std is not None else f"{name}=NA")
        print(f"ODOM_SAMPLES={len(self.odom)}")
        print(f"TRANSFORMED_IMU_SAMPLES={len(self.imu)}")

    def close(self):
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--odom-topic', default='/sea_nav/lio/odom')
    parser.add_argument('--imu-topic', default='/sea_nav/lio/transformed_raw_imu')
    args = parser.parse_args(argv)
    if args.duration <= 0.0:
        parser.error('--duration must be positive')

    import rclpy

    rclpy.init(args=None)
    recorder = StaticRecorder(args)

    def stop(_signum, _frame):
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        recorder.run()
    finally:
        recorder.report()
        recorder.close()


if __name__ == '__main__':
    main()

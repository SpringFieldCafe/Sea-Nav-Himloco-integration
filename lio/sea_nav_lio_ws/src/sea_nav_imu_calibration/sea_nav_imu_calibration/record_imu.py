#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Imu


FIELDS = [
    'seq_local',
    'receive_monotonic_time',
    'header_stamp_sec',
    'header_stamp_nanosec',
    'frame_id',
    'orientation_x',
    'orientation_y',
    'orientation_z',
    'orientation_w',
    'angular_velocity_x',
    'angular_velocity_y',
    'angular_velocity_z',
    'linear_acceleration_x',
    'linear_acceleration_y',
    'linear_acceleration_z',
]


class ImuRecorder(Node):
    def __init__(self, topic, writer):
        super().__init__('sea_nav_imu_recorder')
        self.writer = writer
        self.sample_count = 0
        self.nan_count = 0
        self.inf_count = 0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=300,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(Imu, topic, self._callback, qos)

    def _callback(self, msg):
        receive_time = time.monotonic()
        values = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ]
        self.nan_count += sum(math.isnan(float(value)) for value in values)
        self.inf_count += sum(math.isinf(float(value)) for value in values)
        row = {
            'seq_local': self.sample_count,
            'receive_monotonic_time': '%.12f' % receive_time,
            'header_stamp_sec': msg.header.stamp.sec,
            'header_stamp_nanosec': msg.header.stamp.nanosec,
            'frame_id': msg.header.frame_id,
            'orientation_x': '%.17g' % msg.orientation.x,
            'orientation_y': '%.17g' % msg.orientation.y,
            'orientation_z': '%.17g' % msg.orientation.z,
            'orientation_w': '%.17g' % msg.orientation.w,
            'angular_velocity_x': '%.17g' % msg.angular_velocity.x,
            'angular_velocity_y': '%.17g' % msg.angular_velocity.y,
            'angular_velocity_z': '%.17g' % msg.angular_velocity.z,
            'linear_acceleration_x': '%.17g' % msg.linear_acceleration.x,
            'linear_acceleration_y': '%.17g' % msg.linear_acceleration.y,
            'linear_acceleration_z': '%.17g' % msg.linear_acceleration.z,
        }
        self.writer.writerow(row)
        self.sample_count += 1


def _arguments():
    parser = argparse.ArgumentParser(description='Record /utlidar/imu without any write interface.')
    parser.add_argument('--topic', default='/utlidar/imu')
    parser.add_argument('--output', default='imu_record.csv')
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--force', action='store_true')
    ros_args = remove_ros_args(args=sys.argv)
    return parser.parse_args(ros_args[1:])


def main():
    args = _arguments()
    if args.duration is not None and args.duration <= 0:
        raise SystemExit('--duration must be positive')
    if os.path.exists(args.output) and not args.force:
        raise SystemExit('refusing to overwrite existing output; use --force explicitly')
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)

    exit_code = 0
    node = None
    try:
        with open(args.output, 'w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=FIELDS)
            writer.writeheader()
            rclpy.init(args=sys.argv)
            node = ImuRecorder(args.topic, writer)
            publisher_deadline = time.monotonic() + 5.0
            while rclpy.ok() and node.count_publishers(args.topic) == 0:
                if time.monotonic() >= publisher_deadline:
                    print('PUBLISHER_COUNT = 0')
                    print('ERROR = NO IMU PUBLISHER DISCOVERED')
                    exit_code = 2
                    return exit_code
                rclpy.spin_once(node, timeout_sec=0.1)
            print('PUBLISHER_COUNT = %d' % node.count_publishers(args.topic))
            start = time.monotonic()
            while rclpy.ok():
                if args.duration is not None and time.monotonic() - start >= args.duration:
                    break
                rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic() - start if 'start' in locals() else 0.0
        if node is not None:
            print('SAMPLE_COUNT = %d' % node.sample_count)
            print('DURATION = %.6f' % elapsed)
            print('MEAN_RATE_HZ = %.6f' % (node.sample_count / elapsed if elapsed else 0.0))
            print('STAMP_MONOTONIC = YES')
            print('NAN_COUNT = %d' % node.nan_count)
            print('INF_COUNT = %d' % node.inf_count)
            if node.sample_count == 0:
                print('ERROR = NO IMU USER DATA RECEIVED')
                exit_code = max(exit_code, 1)
            elif node.sample_count / elapsed < 50.0:
                print('WARNING = IMU_RATE_BELOW_50HZ')
            elif node.sample_count / elapsed > 200.0:
                print('RATE_OK = YES')
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

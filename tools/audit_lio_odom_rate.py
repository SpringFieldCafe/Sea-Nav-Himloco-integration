#!/usr/bin/env python3
import argparse
import collections
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="/sea_nav/lio/odom")
    p.add_argument("--duration", type=float, default=10.0)
    args = p.parse_args()

    rclpy.init()
    node = rclpy.create_node("audit_lio_odom_rate")

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1000,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )

    stamps = []
    recv_times = []

    def cb(msg):
        s = msg.header.stamp
        stamps.append(s.sec * 1000000000 + s.nanosec)
        recv_times.append(time.monotonic())

    node.create_subscription(Odometry, args.topic, cb, qos)

    infos = node.get_publishers_info_by_topic(args.topic)
    print(f"TOPIC={args.topic}")
    print(f"PUBLISHER_COUNT={len(infos)}")
    for i, info in enumerate(infos, 1):
        print(f"PUBLISHER_{i}=node:{info.node_name} namespace:{info.node_namespace}")

    t0 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t0 < args.duration:
        rclpy.spin_once(node, timeout_sec=0.05)

    elapsed = time.monotonic() - t0
    c = collections.Counter(stamps)

    callbacks = len(stamps)
    unique = len(c)
    max_dup = max(c.values()) if c else 0

    print(f"ELAPSED_S={elapsed:.3f}")
    print(f"CALLBACKS={callbacks}")
    print(f"CALLBACK_RATE_HZ={callbacks / elapsed if elapsed else 0.0:.3f}")
    print(f"UNIQUE_STAMPS={unique}")
    print(f"UNIQUE_STAMP_RATE_HZ={unique / elapsed if elapsed else 0.0:.3f}")
    print(f"MAX_DUPLICATES_PER_STAMP={max_dup}")

    if callbacks >= 2:
        span = recv_times[-1] - recv_times[0]
        print(f"RECEIVE_SPAN_RATE_HZ={(callbacks - 1) / span if span > 0 else 0.0:.3f}")

    if unique >= 2:
        us = sorted(c.keys())
        span = (us[-1] - us[0]) / 1e9
        print(f"HEADER_STAMP_SPAN_RATE_HZ={(unique - 1) / span if span > 0 else 0.0:.3f}")

    if callbacks == 0:
        print("RESULT=NO_MESSAGES")
        print("HINT=Check Point-LIO is running and source the same Humble/DDS environment in this terminal.")
    elif max_dup > 1:
        print("RESULT=DUPLICATE_HEADER_STAMPS_PRESENT")
    else:
        print("RESULT=OK_NO_DUPLICATE_STAMPS")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Offline one-way IPC fixture sender; it never imports ROS or Torch."""

import argparse
import socket
import time

import numpy as np

from .ipc_schema import encode_packet, make_packet


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/tmp/sea_nav_shadow.sock")
    parser.add_argument("--packets", type=int, default=5)
    args = parser.parse_args(argv)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.socket)
    for sequence in range(args.packets):
        packet = make_packet(
            sequence=sequence, timestamp_monotonic=time.monotonic(), timestamp_wall=time.time(),
            joint_pos=[0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
            joint_vel=np.zeros(12), imu_ang_vel=np.zeros(3), projected_gravity=[0.0, 0.0, -1.0],
            base_linear_velocity_body=np.zeros(3), base_angular_velocity_body=np.zeros(3),
            lidar_rays=np.full(41, 5.0), goal_body=[2.0, 0.0],
            sensor_age={"lowstate": 0.01, "lidar": 0.01, "odom": 0.01, "wireless": None},
            validity={"lowstate": True, "lidar": True, "odom": True, "goal": True},
        )
        sock.sendall(encode_packet(packet)); time.sleep(0.02)
    sock.close(); print(f"fixture_sent={args.packets}")


if __name__ == "__main__":
    main()

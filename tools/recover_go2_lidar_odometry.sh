#!/usr/bin/env bash
set -euo pipefail

# Read-only ROS probe plus the two non-motion Unitree sensor-service switches.
# This script never imports or creates a motion publisher.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NET="${1:-enp3s0}"
PYTHON="${UNITREE_SDK_PYTHON:-/home/hyz/anaconda3/envs/himloco/bin/python}"
LOG_DIR="$ROOT_DIR/logs/go2_service_recovery"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"

exec > >(tee "$LOG_FILE") 2>&1

echo "[recovery] net=$NET"
echo "[recovery] log=$LOG_FILE"
echo "[safety] allowed service switches: unitree_lidar, unitree_lidar_slam"
echo "[safety] motion services are not touched; no LowCmd publisher is created"

"$PYTHON" - "$NET" <<'PY'
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

net = sys.argv[1]
ChannelFactoryInitialize(0, net)
client = RobotStateClient()
client.SetTimeout(5.0)
client.Init()

ALLOWED = {"unitree_lidar", "unitree_lidar_slam"}
WATCH = ALLOWED | {"mcf", "sport_mode", "utrack", "voxel_height_mapping"}

def services(label):
    code, rows = client.ServiceList()
    print("SERVICE_LIST", label, "code=", code)
    current = {}
    for row in rows or []:
        if row.name in WATCH:
            current[row.name] = (row.status, row.protect)
            print(" ", row.name, "status=", row.status, "protect=", row.protect)
    return code, current

services("before")
for name in ("unitree_lidar", "unitree_lidar_slam"):
    if name not in ALLOWED:
        raise RuntimeError("unexpected service name")
    result = client.ServiceSwitch(name, True)
    print("SERVICE_SWITCH", name, "return=", result)
    time.sleep(2.0)
    services("after_" + name)
PY

set +u
source /opt/ros/humble/setup.bash
source /home/hyz/unitree_msgs_humble_ws/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="<CycloneDDS><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"$NET\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

/usr/bin/python3 - "$LOG_FILE" <<'PY'
import math
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2

TOPICS = {
    "RAW_CLOUD": "/utlidar/cloud",
    "RAW_IMU": "/utlidar/imu",
    "CLOUD_BASE": "/utlidar/cloud_base",
    "ROBOT_ODOM": "/utlidar/robot_odom",
}

class Probe(Node):
    def __init__(self):
        super().__init__("go2_lidar_odometry_recovery_probe")
        self.counts = {key: 0 for key in TOPICS}
        self.odom_frames = []
        self.create_subscription(PointCloud2, TOPICS["RAW_CLOUD"], self._raw_cloud, 10)
        self.create_subscription(Imu, TOPICS["RAW_IMU"], self._raw_imu, 10)
        self.create_subscription(PointCloud2, TOPICS["CLOUD_BASE"], self._cloud_base, 10)
        self.create_subscription(Odometry, TOPICS["ROBOT_ODOM"], self._odom, 10)

    def _raw_cloud(self, _msg):
        self.counts["RAW_CLOUD"] += 1

    def _raw_imu(self, _msg):
        self.counts["RAW_IMU"] += 1

    def _cloud_base(self, _msg):
        self.counts["CLOUD_BASE"] += 1

    def _odom(self, msg):
        self.counts["ROBOT_ODOM"] += 1
        self.odom_frames.append((msg.header.frame_id, msg.child_frame_id))

rclpy.init()
node = Probe()
deadline = time.monotonic() + 8.0
while rclpy.ok() and time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.1)

for key, topic in TOPICS.items():
    count = node.counts[key]
    print("%s=%d rate_hz=%.3f" % (key, count, count / 8.0))

frames = [item for item in node.odom_frames if item[0] or item[1]]
odom_frame = frames[-1][0] if frames else ""
child_frame = frames[-1][1] if frames else ""
print("ODOM_FRAME=%s" % odom_frame)
print("ODOM_CHILD_FRAME=%s" % child_frame)
print("ODOM_FRAME_VALID=%s" % str(odom_frame == "odom" and child_frame == "base_link").upper())

result = all(node.counts[key] > 0 for key in TOPICS)
print("RAW_LIDAR_STATUS=%s" % ("PASS" if node.counts["RAW_CLOUD"] > 0 and node.counts["RAW_IMU"] > 0 else "FAIL"))
print("DERIVED_STATUS=%s" % ("PASS" if node.counts["CLOUD_BASE"] > 0 and node.counts["ROBOT_ODOM"] > 0 else "FAIL"))
print("RESULT=%s" % ("PASS" if result and odom_frame == "odom" and child_frame == "base_link" else "FAIL"))
node.destroy_node()
rclpy.shutdown()
PY

echo "[recovery] complete; log=$LOG_FILE"

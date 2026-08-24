#!/usr/bin/env bash
set -e
o pipefail

ROOT=/home/hyz/桌面/sea_nav
LIO_WS=$ROOT/lio/sea_nav_lio_ws

echo "========== GO2 MORNING HIMLOCO + LIO TEST =========="

echo "[1] CPU governor"
cpupower frequency-info | grep "current policy" || true

source /opt/ros/humble/setup.bash
source /home/hyz/unitree_msgs_humble_ws/install/setup.bash
source $LIO_WS/install_humble_clean/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

echo "[2] RAW SENSOR"
timeout 5s ros2 topic hz /utlidar/cloud || true
timeout 5s ros2 topic hz /utlidar/imu || true

echo "[3] START LIO"
LOG=/tmp/go2_morning_lio.log
rm -f $LOG

ros2 launch sea_nav_lio_bringup point_lio_go2.launch.py deskew:=true > $LOG 2>&1 &
LIO_PID=$!

echo "LIO PID=$LIO_PID"

sleep 20

echo "[4] NODES"
ros2 node list | grep -E "point|deskew|transform" || true

echo "[5] ODOM"
timeout 5s ros2 topic hz /sea_nav/lio/odom || true

echo "[6] STATE"
grep '\[LIO-DIAG\] STATE ' $LOG | tail -1 || true

echo "[7] DESKEW"
grep DESKEW_STATUS $LOG | tail -5 || true

echo
echo "LIO CHECK FINISHED"
echo "If state is normal, run HIMLoco manually:"
echo
echo "cd ~/桌面/sea_nav"
echo "conda activate himloco"
echo "python -m deploy.go2_onboard.himloco_fixed_control enp3s0 --policy models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt --vx 0.1 --vy 0 --wz 0 --max-sensor-age 0.10"


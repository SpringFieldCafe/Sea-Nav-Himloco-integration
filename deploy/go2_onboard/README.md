# Go2 onboard sensor/shadow runtime

This milestone is intentionally read-only. Both modes subscribe to Go2 state,
LiDAR, odometry, wireless-controller, and optional `geometry_msgs/PointStamped`
Goal2D topics. `sensor` does not load a policy. `shadow` runs SEA-Nav and
HIMLoco inference but has no `LowCmd` import, publisher, sport-mode call, or
motor-command write path.

Run from the repository root:

```bash
python -m deploy.go2_onboard.runtime \
  --mode sensor \
  --lowstate-topic /lowstate \
  --lidar-topic /utlidar/cloud_base \
  --odom-topic /utlidar/robot_odom \
  --wireless-topic /wirelesscontroller \
  --goal-topic /sea_nav/goal2d \
  --log logs/go2_sensor.jsonl
```

Shadow mode, still with all low-level output disabled:

```bash
python -m deploy.go2_onboard.runtime \
  --mode shadow \
  --navigation-policy artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt \
  --navigation-metadata artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json \
  --himloco-policy models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt \
  --lowstate-topic /lowstate \
  --lidar-topic /utlidar/cloud_base \
  --odom-topic /utlidar/robot_odom \
  --wireless-topic /wirelesscontroller \
  --goal-topic /sea_nav/goal2d \
  --command-filter-alpha 1.0 \
  --device cpu \
  --log logs/go2_shadow.jsonl
```

Goal2D uses `geometry_msgs/PointStamped`: `header.frame_id`, `point.x`,
`point.y`, and the ROS header timestamp. An `odom` goal is transformed to the
robot body frame using odometry position and IMU yaw. A `base_link` goal is
already interpreted as body-frame coordinates. The navigation history is
oldest-to-newest (10 x 55 = 550), while HIMLoco history follows its deployment
contract with the newest frame first (6 x 45 = 270).

The current ROS message types are `unitree_go/LowState`,
`sensor_msgs/PointCloud2`, `nav_msgs/Odometry`,
`unitree_go/WirelessController`, and `geometry_msgs/PointStamped`. Runtime
frequency, age, source timestamp, and frame IDs are measured and logged; the
repository does not hard-code a sensor frequency. The LiDAR adapter assumes
`/utlidar/cloud_base` uses `x=forward, y=left, z=up`; verify this on the Go2
before relying on the shadow output.

The two actual policies used by the documented command are the peer SEA-Nav
TorchScript export and `himloco_himppo_continuous_turning_policy_1460.pt`.
No command publisher is created in either mode, and `send_low_level()` is a
hard-failing test boundary rather than a hidden arm path.

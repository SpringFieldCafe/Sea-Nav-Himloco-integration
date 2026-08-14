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

## Field diagnostics (read-only)

The field diagnostics entry point subscribes to the same four Go2 streams and
prints one bounded JSON record per report interval. It adds the runtime message
type, measured inter-arrival period, maximum gap, stale transition count, raw
odometry fields, LiDAR point validity/range statistics, the 41-ray output, and
all readable `WirelessController` fields. It still has no publisher and sets
`lowcmd_sent` to `false` in every record.

```bash
python -m deploy.go2_onboard.diagnostics \
  --duration 60 \
  --report-interval 1 \
  --max-sensor-age 0.25 \
  --log logs/go2_field_diagnostics.jsonl
```

The `sensor` mode remains the smaller health-only check. Run both on the robot
with the robot stationary before trying shadow inference:

```bash
python -m deploy.go2_onboard.runtime \
  --mode sensor \
  --duration 60 \
  --log logs/go2_sensor_60s.jsonl
```

The diagnostics record intentionally labels odometry twist semantics as
`UNVERIFIED`. Compare `header.frame_id`, `child_frame_id`, and the raw twist
while the robot is stationary, moving straight, and being yawed by hand. Do
not assume that `twist` is body-frame velocity from the topic name alone.

For a stationary goal transform check, publish goals only to the Goal2D input;
this is not a robot command:

```bash
ros2 topic pub --once /sea_nav/goal2d geometry_msgs/msg/PointStamped \
  "{header: {frame_id: odom}, point: {x: 1.0, y: 0.0, z: 0.0}}"
ros2 topic pub --once /sea_nav/goal2d geometry_msgs/msg/PointStamped \
  "{header: {frame_id: odom}, point: {x: 0.0, y: 1.0, z: 0.0}}"
ros2 topic pub --once /sea_nav/goal2d geometry_msgs/msg/PointStamped \
  "{header: {frame_id: odom}, point: {x: 0.0, y: -1.0, z: 0.0}}"
```

Use the diagnostics output, not guessed bit masks, for the wireless-controller
test. Record the `wireless_raw_fields` for no key, A/B/X/Y, Start/Select, and
L1/L2/R1/R2. No arm/stop mapping is accepted by this milestone.

## Shadow timing and filter semantics

`command_filter_alpha` uses
`filtered = alpha * new + (1 - alpha) * old` after the first sample. Thus
`alpha=1.0` means no smoothing and `alpha=0.15` means strong smoothing. The
current read-only runtime default remains `1.0`; use `--command-filter-alpha
0.15` for a like-for-like continuous-turning shadow comparison, without
changing the code default. The first command initializes the filter directly,
so it is not blended with zero.

The runtime prints and logs SEA/HIM inference latency, loop latency, deadline
misses, sensor age, observation finite/min/max, commands, actions, and model
hashes. On the field computer, collect hardware facts separately:

```bash
cat /proc/device-tree/model
lscpu
grep -E 'MemTotal|MemAvailable' /proc/meminfo
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")'
```

`himloco_fixed` is deliberately not exposed in this milestone. Before it can
exist, the wireless mapping, fresh-state gate, explicit arm gate, 50 Hz health
gate, conflict check, and exception/timeout shutdown path must be reviewed in a
separate change. No real command path is present here.

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

## SEA-Nav plus HIMLoco 1460 navigation entry

The first navigation entry keeps the existing guarded HIMLoco low-level
controller and starts SEA-Nav in a separate process. The ROS sensor bridge is
still the only process that reads ROS/DDS sensors. The navigation worker reads
the one-way Unix socket at a low rate, applies the first-milestone limits
(`vx=[0,0.15]`, `vy=0`, `wz=[-0.15,0.15]`), and publishes only the latest command
to the 50 Hz HIMLoco loop. A stale worker result is replaced with zero; the
last non-zero command is never held indefinitely.

The entry requires the approved HIMLoco 1460 hash and the approved peer SEA-Nav
550->3 export. It requires zero fixed-command placeholders because navigation
is the only command source:

Terminal A, with ROS 2 Humble and the existing read-only sensor bridge:

```bash
cd /home/hyz/桌面/sea_nav
source /opt/ros/humble/setup.bash
source /home/hyz/unitree_msgs_humble_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="enp3s0" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

/usr/bin/python3 -m deploy.go2_onboard.sensor_bridge \
  --duration 15 --control-hz 50 --summary-interval 1 \
  --lowstate-max-age 0.10 --odom-max-age 0.10 --lidar-max-age 0.20 \
  --goal-topic '' --goal-frame odom \
  --goal-x GOAL_X --goal-y GOAL_Y \
  --socket /tmp/sea_nav_shadow.sock \
  --log logs/go2_navigation/sensor_bridge.jsonl
```

Terminal B, started after Terminal A prints `listening`:

```bash
cd /home/hyz/桌面/sea_nav
conda activate himloco

python -m deploy.go2_onboard.sea_nav_himloco_navigation \
  enp3s0 \
  --policy models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt \
  --vx 0 --vy 0 --wz 0 \
  --goal-x GOAL_X --goal-y GOAL_Y \
  --sensor-socket /tmp/sea_nav_shadow.sock \
  --navigation-policy artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt \
  --navigation-metadata artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json \
  --navigation-hz 10 \
  --navigation-command-max-age 0.25 \
  --navigation-filter-alpha 0.15 \
  --max-sensor-age 0.10 \
  --navigation-log logs/go2_navigation/navigation.jsonl \
  --event-trace /tmp/sea_nav_navigation_event_trace.jsonl
```

`GOAL_X` and `GOAL_Y` are placeholders for a goal already expressed in the
`odom` frame. This command is documentation only; the navigation controller
has not been run on a real Go2 in this milestone. The normal Start -> pose
transition -> default-pose hold -> A arm sequence, performance-governor gate,
motion-owner check, wireless STOP/ESTOP, LowState watchdog, and Ctrl+C stop
path remain owned by `himloco_fixed_control.py`.

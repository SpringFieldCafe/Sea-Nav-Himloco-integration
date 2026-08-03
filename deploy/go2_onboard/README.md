# Go2 onboard shadow runtime

This runtime is intentionally read-only. It subscribes to Go2 state, LiDAR,
odometry, and wireless-controller topics, runs the SEA-Nav navigation actor and
HIMLoco actor, and prints diagnostics. It does not create `/lowcmd`, call a
sport-mode API, or send any motor command.

Run from the repository root:

```bash
python -m deploy.go2_onboard.shadow_runtime \
  --navigation-policy /home/unitree/sea_nav_models/navigation/sea_nav_policy_2000.pt \
  --navigation-metadata /home/unitree/sea_nav_models/navigation/sea_nav_policy_2000.json \
  --himloco-policy /home/unitree/sea_nav_models/locomotion/himloco/policy_1.pt \
  --lowstate-topic /lowstate \
  --lidar-topic /utlidar/cloud_base \
  --odom-topic /utlidar/robot_odom \
  --wireless-topic /wirelesscontroller \
  --device cuda
```

The navigation goal is supplied as a fixed body-frame target for this shadow
version with `--goal-x` and `--goal-y`; it does not implement a global map or
goal service. The navigation history is oldest-to-newest (10 x 55 = 550),
while the HIMLoco history follows its deployment contract with the newest
frame first (6 x 45 = 270).

The ROS message packages and point-cloud coordinate frame must be verified on
the Go2 before running. The LiDAR adapter assumes `/utlidar/cloud_base` uses
`x=forward, y=left, z=up`.

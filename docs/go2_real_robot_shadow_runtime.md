# Go2 真机 Sensor / Shadow Runtime

当前分支 `feature/go2-real-robot-validation` 只包含真机前的只读验证层。
`sensor` 和 `shadow` 两种模式都不导入 `LowCmd`、不创建 publisher、不调用
运动模式接口，也不会发送任何关节命令。

## 已审计接口

| 数据 | ROS 类型 | 默认 topic | 运行时记录 |
| --- | --- | --- | --- |
| 机器人状态、IMU、关节 q/dq | `unitree_go/LowState` | `/lowstate` | age、频率、frame、有限性 |
| LiDAR | `sensor_msgs/PointCloud2` | `/utlidar/cloud_base` | age、频率、frame、点数、有限性 |
| 里程计 | `nav_msgs/Odometry` | `/utlidar/robot_odom` | age、频率、header frame、child frame |
| 无线控制器 | `unitree_go/WirelessController` | `/wirelesscontroller` | age、频率、显式急停字段 |
| 目标 | `geometry_msgs/PointStamped` | `/sea_nav/goal2d` | frame、x、y、ROS timestamp |

LiDAR 适配器沿用训练合同：body frame 为 `x=forward, y=left, z=up`，41 条
射线，角度范围 `[-2*pi/3, 2*pi/3]`，距离裁剪 `[0.1, 5.0]`，最后做
`log2`。真实设备的 frame 必须在运行前确认，代码不会猜测或自动纠正 frame。

## 模型合同

SEA-Nav 每帧 55 维、10 帧 `oldest_to_newest`：

```text
projected_gravity              3, scale 1.0
navigation_command_scaled      3, scale [2.0, 2.0, 0.25]
base_linear_velocity           3, scale 1.0
base_angular_velocity          3, scale 1.0
ray_distance_log2             41, clip [0.1, 5.0]
relative_goal_xy               2, robot body frame
```

HIMLoco 每帧 45 维、6 帧 `newest_to_oldest`，输入 270、输出 12：

```text
command_scaled 3, [2.0, 2.0, 0.25]
angular_velocity * 0.25       3
projected_gravity              3
(q - default_q) * 1.0        12
dq * 0.05                    12
previous_action               12, clipped [-100, 100]
```

关节 policy 顺序是 `FL, FR, RL, RR`，每条腿为 `hip, thigh, calf`。Go2
LowState motor order 通过 `deploy/go2_onboard/joint_mapping.py` 显式转换，
不会按数组位置直接假定相同顺序。

## 运行命令

只读传感器检查，不加载模型：

```bash
cd /home/hyz/桌面/sea_nav
conda activate himloco
python -m deploy.go2_onboard.runtime \
  --mode sensor \
  --lowstate-topic /lowstate \
  --lidar-topic /utlidar/cloud_base \
  --odom-topic /utlidar/robot_odom \
  --wireless-topic /wirelesscontroller \
  --goal-topic /sea_nav/goal2d \
  --log logs/go2_sensor.jsonl
```

完整策略 shadow，不输出真实 low-level command：

```bash
cd /home/hyz/桌面/sea_nav
conda activate himloco
python -m deploy.go2_onboard.runtime \
  --mode shadow \
  --navigation-policy artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt \
  --navigation-metadata artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json \
  --himloco-policy models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt \
  --goal-topic /sea_nav/goal2d \
  --command-filter-alpha 1.0 \
  --device cpu \
  --log logs/go2_shadow.jsonl
```

为了离线检查目标变换，也可以使用一次性 body/odom 目标参数；这仍然是
shadow，不会发布命令：

```bash
python -m deploy.go2_onboard.runtime \
  --mode shadow \
  --navigation-policy artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt \
  --navigation-metadata artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json \
  --himloco-policy models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt \
  --goal-topic '' --goal-frame odom --goal-x 2.0 --goal-y 0.0 \
  --duration 10 --log logs/go2_shadow_offline_goal.jsonl
```

## 当前缺口与风险

- ROS 话题存在性、真实频率和 frame 仍需在 Go2 上通过 `ros2 topic info`
  和 `ros2 topic echo` 确认。
- `LowState` 当前只提供 IMU quaternion/gyro 和 q/dq；其消息本身没有统一
  header frame，日志会保留空 frame，不会伪造 frame。
- `Odometry.twist` 必须确认是在 `base_link`（或 child frame）表达；不满足
  时不能把它直接当作训练所需 body-frame velocity。
- 无线急停只识别消息中明确的布尔字段
  `emergency_stop`、`estop`、`e_stop`，不会猜测 `keys` 位编码。真实机测试
  前必须核对 Unitree 消息定义并单独完成急停映射。
- 仍缺少真实 Go2 上的传感器数据回放和端到端 shadow 验收；本阶段没有
  `himloco_fixed` 模式，也没有真实运动测试。

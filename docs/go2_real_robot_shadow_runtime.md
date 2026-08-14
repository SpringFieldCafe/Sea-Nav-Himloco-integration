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

## 阶段 3A/3B 现场采集流程

代码仓库无法替代真实 Go2 现场数据，因此以下项目必须在机器旁完成并把
JSONL 保存下来。所有命令都是订阅或打印，不会发布 LowCmd。

### 1. Topic、频率、frame、dropout

先查看实际类型和连接情况：

```bash
ros2 topic list | grep -E 'lowstate|cloud_base|robot_odom|wirelesscontroller'
ros2 topic info /lowstate -v
ros2 topic info /utlidar/cloud_base -v
ros2 topic info /utlidar/robot_odom -v
ros2 topic info /wirelesscontroller -v
```

再连续采集至少 60 秒：

```bash
python -m deploy.go2_onboard.diagnostics \
  --duration 60 \
  --report-interval 1 \
  --max-sensor-age 0.25 \
  --log logs/go2_field_diagnostics_60s.jsonl
```

记录中的 `sensor.streams.*` 包含实际 message type、收到消息的 age、最近
频率、最近间隔、最大间隔、stale transition count、header frame 和 source
timestamp。LowState 若没有 ROS header，frame 为空是预期行为，不应人为填入。

### 2. Odom twist 语义实验

诊断日志会同时保留：

```text
header_frame_id
child_frame_id
position_xyz_m
orientation_wxyz
twist_linear_raw
twist_angular_raw
twist_frame_semantics = UNVERIFIED
```

按以下顺序操作且保持机器人不使能：静止 10 秒、人工沿机身前方移动、人工
改变 yaw 方向但不让机器人行走。用 position/yaw 与 twist 的方向和符号对照，
确认 twist 属于 odom/world 还是 child/body frame。当前 runtime 把 odom
linear velocity 直接送入 SEA-Nav；如果现场证明它不是 body-frame，必须在下一
个审查变更中修正，当前代码不会偷偷转换。

### 3. LiDAR 合同现场统计

诊断记录同时报告原始点数量、有限点数量、NaN/Inf 数量、有效 xy range、
接近原点点数，以及实际送入 adapter 的点数。adapter 的固定合同是：

```text
x=forward, y=left, z=up       # 必须现场验证
41 rays
angle = [-2*pi/3, +2*pi/3]
distance clip = [0.1, 5.0] m
policy value = log2(distance)
```

`rays_41`、`log2_rays_min/max` 会写入每条诊断记录。接近原点点数只能提示
机器人自身点的可能性，不能单独证明点的来源；应在静止时遮挡/移开障碍物并
对照点云 frame 做最终确认。

### 4. Goal2D 静止验证

目标输入使用 `geometry_msgs/PointStamped`，不属于机器人运动命令：

```bash
ros2 topic pub --once /sea_nav/goal2d geometry_msgs/msg/PointStamped \
  "{header: {frame_id: odom}, point: {x: 1.0, y: 0.0, z: 0.0}}"
ros2 topic pub --once /sea_nav/goal2d geometry_msgs/msg/PointStamped \
  "{header: {frame_id: odom}, point: {x: 0.0, y: 1.0, z: 0.0}}"
ros2 topic pub --once /sea_nav/goal2d geometry_msgs/msg/PointStamped \
  "{header: {frame_id: odom}, point: {x: 0.0, y: -1.0, z: 0.0}}"
```

在日志中检查 `goal_robot_xy` 的前/左/右符号。不要发布 `cmd_vel`、LowCmd
或任何运动模式消息。

### 5. Shadow 性能采集

```bash
python -m deploy.go2_onboard.runtime \
  --mode shadow \
  --navigation-policy artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt \
  --navigation-metadata artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json \
  --himloco-policy models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt \
  --command-filter-alpha 0.15 \
  --device cpu \
  --duration 60 \
  --log logs/go2_shadow_60s.jsonl
```

日志中的 `sea_inference_latency_ms`、`him_inference_latency_ms` 和
`loop_latency_ms` 是逐周期样本；现场可用离线脚本计算 mean/P50/P95/P99/max。
`deadline_miss`、sensor age、obs finite/min/max、raw/filtered command 和
action 也会逐周期保存。当前默认 device 是 CPU；只有现场明确安装并验证
CUDA/PyTorch 后才使用 `--device cuda`。

`command_filter_alpha` 的实际公式是：

```text
filtered = alpha * new + (1 - alpha) * old
```

因此：`1.0` = 无滤波，`0.15` = 15% 新命令 + 85% 旧命令。当前代码默认值
保持 `1.0`，没有擅自改动；连续转弯对照应显式传 `0.15`。首次收到命令时
直接初始化为新命令，不与零混合，这一点应纳入现场合同记录。

### 6. WirelessController 原始字段

```bash
python -m deploy.go2_onboard.diagnostics \
  --duration 60 \
  --report-interval 0.2 \
  --log logs/go2_wireless_raw_60s.jsonl
```

在无按键、A/B/X/Y、Start/Select、L1/L2/R1/R2 各状态下分别保存
`wireless_raw_fields`。当前只识别消息中明确的布尔字段
`emergency_stop`、`estop`、`e_stop`；不会猜 Unitree 的 bit mask，也不会基于
猜测实现 ARM/STOP。

## 硬件记录

在 Go2 NVIDIA 电脑上执行并保存输出：

```bash
cat /proc/device-tree/model
lscpu
grep -E 'MemTotal|MemAvailable' /proc/meminfo
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")'
```

最终报告需要把这些硬件信息与 shadow JSONL 的延迟分位数一起审查，重点是
50 Hz 的 20 ms deadline 是否有余量。

## 阶段 3C 安全设计（尚未实现/尚未执行）

本分支当前没有 `himloco_fixed`。在开放任何真实运动前，独立变更至少需要：

```text
INIT
  -> SENSOR_WAIT
  -> SENSOR_READY
  -> ARMED_FIXED_COMMAND
  -> RUNNING_FIXED_COMMAND
  -> STOPPING
  -> ESTOP
```

进入 `ARMED_FIXED_COMMAND` 必须同时满足 explicit arm、fresh LowState、有效
IMU/q/dq、有限模型输出、模型 hash/shape 合同、50 Hz deadline、无线 STOP/ESTOP
映射和 controller conflict check。任何 Ctrl+C、异常、传感器超时、NaN/Inf 或
deadline 连续失败都必须走 `STOPPING`/`ESTOP`。第一版固定命令还必须使用配置
中的小范围上限，并配 publisher mock/no-write 回归测试。本阶段不实现、不运行、
不发送任何真实控制命令。

# Go2 SEA-Nav 真机导航验证阶段性汇报

> 分支：`experiment/go2-seanav-himloco-2mps`
> 工作目录：`/home/hyz/桌面/sea_nav`
> 更新时间：2026-08-25

## 1. 项目目标

本阶段目标是形成一条傻瓜式、一键启动、可记录日志、可安全退出的 Go2 真机导航验证流程：启动前清理旧进程，检查 CPU/MCF/传感器，启动官方雷达和 SEA-Nav 导航，执行目标点运动，并在退出时统一停止相关进程。

当前入口脚本：

```bash
bash tools/go2_seanav_navigation_test.sh
```

## 2. 当前推荐架构

目前推荐使用“官方运动控制 + 官方雷达链路 + SEA-Nav 导航策略”，而不是 HIMLoco 直接控制底层步态的路径。

```text
Go2 官方雷达/里程计
        │
        ▼
sensor_bridge ── Unix socket ──> SEA-Nav navigation
                                      │
                                      ▼
                         Unitree SportClient.Move(vx, vy, vyaw)
                                      │
                                      ▼
                              官方 Sport/MPC 步态
```

官方路径正常时，摘要应接近：

```text
CPU=PASS MCF=PASS SENSOR=PASS
TRANSFORM=OFFICIAL_NATIVE DESKEW=OFFICIAL_NATIVE
POINT_LIO=NOT_USED_OFFICIAL NAVIGATION=RUNNING
```

官方路径不启动 Point-LIO，也不使用 HIMLoco policy 直接输出运动控制指令；实际运动由官方 `SportClient.Move` 完成。

## 3. 推荐一键启动命令

### 3.1 前进 2 米、右侧 1 米的保守测试

此前出现过目标点后继续横向运动、绕圈和碰撞障碍物的现象，所以首次复测建议限制速度、不允许倒退，并保留 0.30 米停车容差：

```bash
cd /home/hyz/桌面/sea_nav

sudo cpupower frequency-set -g performance

bash tools/go2_seanav_navigation_test.sh \
  --front-goal-forward 2.0 \
  --front-goal-left -1.0 \
  --navigation-vx-min 0 \
  --navigation-vx-max 0.60 \
  --navigation-vy-max 0.20 \
  --goal-tolerance 0.30 \
  --goal-slowdown-distance 0.60 \
  --enable-official-motion
```

首次测试建议不要加 `--non-interactive`。看到红色 `START` 提示后，确认机器狗站稳、周围没有人员和障碍物，再输入：

```text
START
```

确认流程稳定后，才可以在自动化测试中追加：

```bash
--non-interactive
```

当前约定 `x` 为前方、`y` 为左方，因此右侧 1 米使用 `--front-goal-left -1.0`。

### 3.2 仅前进 2 米

```bash
bash tools/go2_seanav_navigation_test.sh \
  --front-goal-distance 2.0 \
  --navigation-vx-min 0 \
  --navigation-vx-max 0.60 \
  --navigation-vy-max 0.20 \
  --goal-tolerance 0.30 \
  --goal-slowdown-distance 0.60 \
  --enable-official-motion
```

`--front-goal-distance` 不能与 `--front-goal-forward`、`--front-goal-left` 同时使用。

## 4. 参数说明

| 参数 | 含义 | 建议 |
|---|---|---|
| `--front-goal-distance M` | 以首次里程计位姿为基准，在前方设置 M 米目标 | 单方向前进 |
| `--front-goal-forward M` | 前方偏移 M 米 | 与 `--front-goal-left` 配合 |
| `--front-goal-left M` | 左方偏移 M 米，负值表示右方 | 右 1 米用 `-1.0` |
| `--goal-x X --goal-y Y` | 直接指定世界/里程计坐标目标 | 已知固定坐标时使用 |
| `--navigation-vx-max V` | 最大前向速度，m/s | 首测建议 `0.60` |
| `--navigation-vx-min V` | 最小前向速度，负值允许倒退 | 首测建议 `0` |
| `--navigation-vy-max V` | 横向速度上限，m/s | 首测建议 `0.20` |
| `--goal-tolerance M` | 目标停止半径 | 当前建议 `0.30` |
| `--goal-slowdown-distance M` | 距目标 M 米内开始平滑减速 | 当前建议 `0.60` |
| `--enable-official-motion` | 将导航指令传给官方 Sport/MPC | 真机运动必须加 |
| `--non-interactive` | 跳过 `START` 手动确认 | 仅安全自动化测试使用 |
| `--assume-clear-lidar` | 声明区域无障碍，关闭障碍规避假设 | 不建议常规真机使用 |
| `--policy PATH` | HIMLoco 1460 policy | 官方 Sport/MPC 路径不实际使用 |
| `--navigation-policy PATH` | SEA-Nav 导航策略 | 当前官方路径实际使用 |
| `--navigation-metadata PATH` | 导航策略元数据 | 与导航策略配套 |
| `--fixed-sport-vx V` | 固定 SportClient 速度诊断模式 | 仅受控诊断使用 |

完整帮助：

```bash
bash tools/go2_seanav_navigation_test.sh --help
```

## 5. 已完成和已验证的问题修复

### 5.1 一键启动、清理和退出

脚本启动前会清理旧的 SEA-Nav、MCF、传感器桥接、Point-LIO、transform、adapter 和监控进程；退出时按 PID/进程组停止本次启动的进程，避免重复启动。`Ctrl+C`、异常退出和 B 急停都会进入统一清理流程。

### 5.2 切换到官方雷达链路

官方路径使用 `/utlidar/cloud_base`、`/utlidar/cloud_deskewed`、`/utlidar/robot_odom`、官方原生 deskew 和官方里程计，不启动 Point-LIO。此前自检中 IMU 和原始雷达通过，但自定义 adapter 被判定为 unhealthy；切换官方链路后，运行摘要多次出现 `SENSOR=PASS`、`DESKEW=OFFICIAL_NATIVE`、`POINT_LIO=NOT_USED_OFFICIAL`。

### 5.3 MCF 和 CPU 检查

启动前会检查 MCF，必要时请求释放并记录结果。CPU governor 不为 `performance` 时会停止启动，避免性能波动影响实验。可手动执行：

```bash
sudo cpupower frequency-set -g performance
```

曾经出现的 `CPU=FAIL` 属于启动前置检查失败，不是导航算法已运行后的失败。

### 5.4 目标附近减速和参数化

已增加 `--goal-slowdown-distance`，在接近目标时平滑缩放前向速度；`--goal-tolerance` 控制停止半径。当前工作区还加入了可配置的 `--navigation-vx-min`，可以选择是否允许倒退，但这部分修改尚未提交和推送，仍需独立验证。

### 5.5 日志和运行摘要

每次运行都会创建：

```text
logs/go2_seanav_navigation/YYYYMMDD_HHMMSS/
```

重点文件：

| 文件 | 内容 |
|---|---|
| `summary.txt` | 参数、状态、结果和退出原因 |
| `himloco.log` / `navigation.log` | 导航循环、目标距离、速度和异常 |
| `state_diagnostics.jsonl` | 传感器年龄、频率和状态 |
| `odom.jsonl` | 官方里程计输出 |
| `logs/cloud_hz.log` | 官方点云频率 |
| `logs/deskew_hz.log` | 去畸变点云频率 |
| `logs/odom_hz.log` | 官方里程计频率 |
| `logs/motion_metrics.log` | 里程、位移和平均速度 |

## 6. 已发现但尚未彻底解决的问题

### 6.1 到达目标附近后绕圈或继续横向走

日志表明，当目标相对狗体已经位于后方时，如果禁止倒退，控制器只能通过横向移动和转向继续寻找目标，因而可能绕圈、路径变长，甚至向侧方持续运动。

证据包括：

- `logs/go2_seanav_navigation/20260825_143410`：最近点附近 `goal_body≈[-0.28, 0.03]`，目标已在狗体后方；路径约 6.77 米。
- `logs/go2_seanav_navigation/20260825_145510`：最近点约 0.24 米，但最终距离又增大到约 0.94 米，说明到达附近后没有稳定锁存停车。

因此，当前主要问题不应只归结为雷达。更可能的组合原因是目标到达判定、目标锁存、目标在狗体后方时的控制策略，以及横向/转向指令没有独立的硬安全约束。

### 6.2 软件误差与尺量误差不一致

软件误差是控制器参考点（当前为 `base_link`）到目标的误差，不一定等于人工测量的狗头、狗身中心或雷达外壳到目标的距离。此前软件最近距离约 0.13～0.24 米，而人工测量约 0.35～0.40 米，说明需要做物理参考点标定，不能直接把两者当成同一个误差。

### 6.3 坐标正负方向仍需低速现场确认

当前代码约定：

```text
x 正方向 = 狗前方
y 正方向 = 狗左方
y 负方向 = 狗右方
```

官方 `Move(vx, vy, vyaw)` 接口明确使用 x、y、yaw 三个速度量，但接口定义本身没有完整标注 y 的“左/右”文字说明。因此，右向测试应先用低速、短距离、空旷环境验证。

### 6.4 到达后没有明确的任务锁存状态

摘要中的 `NAVIGATION=RUNNING` 只表示导航进程仍在运行，不等价于机器狗已经安全停车。后续应明确区分：

```text
GOAL_REACHED       目标判定满足
MOTION_STOPPED     已连续发送零速并稳定停车
MISSION_FINISHED   任务锁存，不再重新启动运动
```

## 7. 坐标系和官方接口说明

当前代码检查官方里程计为：

```text
header.frame_id = odom
child_frame_id  = base_link
```

目标先在 `odom`/世界平面表达，再根据当前 yaw 转换到 `base_link` 下的 `goal_body`。所以 `goal_body.x < 0` 表示目标在狗体后方，不直接等于雷达坐标系错误。

官方 LiDAR SDK 说明，点云坐标原点位于 LiDAR 底部安装面中心，不是狗的几何中心，也不是狗头最前端。因此，雷达坐标、`base_link` 和人工尺量点不能默认视为同一个物理点。

参考资料：

- [Unitree SDK2 Python SportClient](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/unitree_sdk2py/go2/sport/sport_client.py)
- [Unitree SDK2 Go2 SportClient header](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/go2/sport/sport_client.hpp)
- [Unitree ROS2 官方仓库](https://github.com/unitreerobotics/unitree_ros2)
- [Unitree LiDAR SDK2 坐标说明](https://github.com/unitreerobotics/unilidar_sdk2)

## 8. 后续解决方案

### 第一阶段：安全确认方向

1. 在空旷环境用很低速度验证 `vy=-0.1` 是否确实向右。
2. 在坐标符号确认前，不启用大横向速度和倒退。
3. 保留人员急停和 `Ctrl+C`，不要把 `--assume-clear-lidar` 当作常规安全机制。
4. 目标接近时同时限制前向、横向和角速度，而不是只降低前向速度。

### 第二阶段：修复任务状态机

1. 目标进入容差后进入 `GOAL_LATCHED`。
2. 连续多个周期满足容差后确认到达，避免单帧误判。
3. 到达后持续发送零速，禁止重新规划和再次启动运动。
4. 对 `goal_body.x < 0` 制定明确策略：停车、原地转向，或经过安全验证后受限倒退，不能让横向控制无限持续。

### 第三阶段：物理坐标标定

1. 在机体上标出 `base_link`、LiDAR 原点和人工测量中心。
2. 记录 `/utlidar/robot_odom` 与官方点云坐标。
3. 用已知直线和横向距离标定 x/y 正方向及尺度。
4. 统一目标圆心的验收定义，最终以已标定参考点测量。

### 第四阶段：逐步提高速度

```text
vx=0.30, vy=0.10
        ↓
vx=0.60, vy=0.20
        ↓
确认方向、停车和误差稳定
        ↓
再逐步提高速度上限
```

在坐标方向、目标锁存和安全停车未确认前，不建议直接恢复 `vx=2.0 m/s` 或启用较大的横向速度。

## 9. 如何判断一次实验真正完成

不能只看：

```text
NAVIGATION=RUNNING
RESULT=PASS
```

还要检查 `summary.txt` 和导航日志，确认：

```text
GOAL_REACHED_COUNT > 0
最终 goal_distance 在目标容差内
最终速度接近 0
没有持续横向/旋转指令
没有 stale_sensor、PROCESS_ERROR、B_ESTOP
```

查看最新运行：

```bash
LATEST=$(find /home/hyz/桌面/sea_nav/logs/go2_seanav_navigation \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
echo "$LATEST"
sed -n '1,240p' "$LATEST/summary.txt"
tail -n 120 "$LATEST/himloco.log"
```

## 10. 当前结论

官方雷达 + 官方 deskew/odom + 官方 Sport/MPC 路径已经能够稳定启动，整体明显优于 HIMLoco 直接控制底层步态的路径。当前剩余问题主要不是“能否启动”，而是目标参考点、坐标方向、目标到达状态机和到达后的安全停车没有完全闭环。

现阶段最稳妥的路线是保持官方运动链路，先用低速和受限横向速度完成方向验证，同时修复目标锁存、到达停车和物理参考点标定，再逐步恢复速度上限。

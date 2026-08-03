from dataclasses import dataclass, field
from typing import List


@dataclass
class RuntimeConfig:
    navigation_policy: str
    himloco_policy: str
    navigation_metadata: str = ""
    lowstate_topic: str = "/lowstate"
    lidar_topic: str = "/utlidar/cloud_base"
    odom_topic: str = "/utlidar/robot_odom"
    wireless_topic: str = "/wirelesscontroller"
    device: str = "cpu"
    control_hz: float = 50.0
    max_sensor_age: float = 0.25
    goal_xy: List[float] = field(default_factory=lambda: [0.0, 0.0])
    command: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    max_nav_action: float = 3.0
    command_bounds_min: List[float] = field(default_factory=lambda: [-1.0, -1.0, -2.0])
    command_bounds_max: List[float] = field(default_factory=lambda: [1.0, 1.0, 2.0])
    ray_count: int = 41
    ray_min_distance: float = 0.1
    ray_max_distance: float = 5.0
    ray_angle_min: float = -2.0 * 3.141592653589793 / 3.0
    ray_angle_max: float = 2.0 * 3.141592653589793 / 3.0
    lidar_min_z: float = -0.25
    lidar_max_z: float = 1.0
    log_interval: float = 1.0

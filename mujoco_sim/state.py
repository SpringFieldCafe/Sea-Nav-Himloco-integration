from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class UnifiedState:
    """Numerical state shared by ROS2 and MuJoCo adapters."""

    gravity: np.ndarray
    angular_velocity: np.ndarray
    linear_velocity: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    position_xy: np.ndarray
    rays: np.ndarray
    goal_xy: np.ndarray
    roll: float = 0.0
    pitch: float = 0.0
    terrain_hint: str = "flat"
    min_obstacle_distance: float = 5.0
    collision: bool = False
    fallen: bool = False
    timestamp: float = 0.0
    extras: dict = field(default_factory=dict)

    def validate(self):
        expected = {
            "gravity": (3,), "angular_velocity": (3,), "linear_velocity": (3,),
            "joint_position": (12,), "joint_velocity": (12,),
            "position_xy": (2,), "rays": (41,), "goal_xy": (2,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid unified state {name}: {value.shape}")


@dataclass
class RuntimeOutput:
    raw_command: np.ndarray
    supervised_command: np.ndarray
    himloco_action: np.ndarray
    supervisor_state: str
    nav_latency_ms: float = 0.0
    himloco_latency_ms: float = 0.0
    control_hz: float = 0.0
    goal_reached: bool = False
    diagnostics: Optional[dict] = None

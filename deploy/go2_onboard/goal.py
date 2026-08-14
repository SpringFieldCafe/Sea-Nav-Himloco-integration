"""Frame-aware 2D goal interface for the onboard runtime."""

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Goal2D:
    frame_id: str
    x: float
    y: float
    timestamp: float

    @property
    def xy(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float32)


class GoalManager:
    """Stores the latest goal and converts it into the robot body frame."""

    def __init__(self):
        self._goal: Optional[Goal2D] = None

    @property
    def current(self) -> Optional[Goal2D]:
        return self._goal

    def update(self, goal: Goal2D) -> None:
        if not goal.frame_id:
            raise ValueError("Goal2D.frame_id must not be empty")
        if not all(math.isfinite(float(v)) for v in (goal.x, goal.y, goal.timestamp)):
            raise ValueError("Goal2D contains NaN or Inf")
        self._goal = goal

    def relative_xy(
        self,
        robot_position_xy: Sequence[float],
        robot_yaw: float,
        odom_frame_id: str,
        base_frame_id: str = "base_link",
    ) -> np.ndarray:
        if self._goal is None:
            raise RuntimeError("no Goal2D has been received")
        goal = self._goal
        robot_position_xy = np.asarray(robot_position_xy, dtype=np.float32).reshape(2)
        if not np.isfinite(robot_position_xy).all() or not math.isfinite(float(robot_yaw)):
            raise FloatingPointError("robot pose contains NaN or Inf")
        if goal.frame_id in (base_frame_id, "base", "base_link"):
            return goal.xy
        if not odom_frame_id or goal.frame_id != odom_frame_id:
            raise ValueError(
                f"goal frame {goal.frame_id!r} does not match odometry frame {odom_frame_id!r}"
            )
        delta = goal.xy - robot_position_xy
        c, s = math.cos(float(robot_yaw)), math.sin(float(robot_yaw))
        return np.asarray([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]], dtype=np.float32)

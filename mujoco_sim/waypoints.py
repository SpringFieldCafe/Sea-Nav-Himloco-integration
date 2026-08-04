import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    radius: float = 0.5
    terrain: Optional[str] = None

    @property
    def xy(self):
        return np.asarray([self.x, self.y], dtype=np.float32)


class WaypointManager:
    """Ordered world-frame waypoint queue with stop-on-final-goal semantics."""

    def __init__(self, waypoints: Iterable[Waypoint], default_radius=0.5):
        self.waypoints = tuple(waypoints)
        if not self.waypoints:
            raise ValueError("at least one waypoint is required")
        if any(w.radius <= 0 for w in self.waypoints):
            raise ValueError("waypoint radius must be positive")
        self.default_radius = default_radius
        self.index = 0
        self.reached_indices = []
        self.reached = False

    @classmethod
    def from_json(cls, path, default_radius=0.5):
        with open(path, "r", encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, list):
            raise ValueError("waypoint file must contain a JSON list")
        waypoints = []
        for value in values:
            waypoints.append(Waypoint(
                float(value["x"]), float(value["y"]),
                float(value.get("radius", default_radius)), value.get("terrain")))
        return cls(waypoints, default_radius)

    @classmethod
    def single(cls, x, y, radius=0.5):
        return cls([Waypoint(float(x), float(y), float(radius))], radius)

    @property
    def current(self):
        return self.waypoints[min(self.index, len(self.waypoints) - 1)]

    @property
    def current_index(self):
        return self.index

    @property
    def total(self):
        return len(self.waypoints)

    @property
    def done(self):
        return self.reached

    def update(self, position_xy):
        if self.reached:
            return False
        position_xy = np.asarray(position_xy, dtype=np.float32)
        distance = float(np.linalg.norm(self.current.xy - position_xy))
        if distance > self.current.radius:
            return False
        self.reached_indices.append(self.index)
        if self.index + 1 == len(self.waypoints):
            self.reached = True
        else:
            self.index += 1
        return True

    def distance(self, position_xy):
        return float(np.linalg.norm(self.current.xy - np.asarray(position_xy, dtype=np.float32)))

    def relative_goal(self, position_xy, body_rotation):
        delta = self.current.xy - np.asarray(position_xy, dtype=np.float32)
        return np.asarray(body_rotation, dtype=np.float32).T @ delta

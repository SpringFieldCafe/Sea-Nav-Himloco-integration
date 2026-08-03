import math

import torch

from .history_buffer import HistoryBuffer


class NavigationObservation:
    """Builds the exact SEA-Nav 55D frame and 10-frame 550D history."""

    def __init__(self, device: torch.device):
        self.device = device
        self.history = HistoryBuffer(10, 55, "oldest_to_newest", device)

    def reset(self):
        self.history.reset()

    def build_frame(self, gravity, command, linear_velocity, angular_velocity, rays, goal_xy):
        values = [gravity, command * torch.tensor([2.0, 2.0, 0.25], device=self.device),
                  linear_velocity, angular_velocity, torch.log2(rays.clamp(0.1, 5.0)), goal_xy]
        frame = torch.cat(values, dim=-1).to(device=self.device, dtype=torch.float32)
        if tuple(frame.shape) != (1, 55):
            raise RuntimeError(f"SEA-Nav frame must be (1,55), got {tuple(frame.shape)}")
        if not torch.isfinite(frame).all():
            raise FloatingPointError("SEA-Nav observation contains NaN or Inf")
        return frame

    def build(self, gravity, command, linear_velocity, angular_velocity, rays, goal_xy):
        frame = self.build_frame(gravity, command, linear_velocity, angular_velocity, rays, goal_xy)
        observation = self.history.push(frame, repeat_on_first=True)
        if tuple(observation.shape) != (1, 550):
            raise RuntimeError(f"SEA-Nav observation must be (1,550), got {tuple(observation.shape)}")
        return observation

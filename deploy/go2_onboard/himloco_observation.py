import torch

from .history_buffer import HistoryBuffer


class HIMLocoObservation:
    """Matches HIMLoco Go2 deployment: newest frame first, 45*6=270."""

    def __init__(self, device: torch.device):
        self.device = device
        self.history = HistoryBuffer(6, 45, "newest_to_oldest", device)
        self.previous_action = torch.zeros((1, 12), dtype=torch.float32, device=device)

    def reset(self):
        self.history.reset()
        self.previous_action.zero_()

    def build(self, command, angular_velocity, gravity, joint_position, joint_velocity):
        command_scaled = command * torch.tensor([2.0, 2.0, 0.25], device=self.device)
        q_scaled = joint_position * 1.0
        dq_scaled = joint_velocity * 0.05
        frame = torch.cat(
            (command_scaled, angular_velocity * 0.25, gravity, q_scaled, dq_scaled, self.previous_action),
            dim=-1,
        ).to(device=self.device, dtype=torch.float32)
        if tuple(frame.shape) != (1, 45):
            raise RuntimeError(f"HIMLoco frame must be (1,45), got {tuple(frame.shape)}")
        if not torch.isfinite(frame).all():
            raise FloatingPointError("HIMLoco observation contains NaN or Inf")
        observation = self.history.push(frame)
        if tuple(observation.shape) != (1, 270):
            raise RuntimeError(f"HIMLoco observation must be (1,270), got {tuple(observation.shape)}")
        return observation

    def record_action(self, policy_action):
        self.previous_action = policy_action.detach().clamp(-100.0, 100.0)

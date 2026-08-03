from typing import Optional

import torch


class HistoryBuffer:
    """Small fixed-size tensor history with an explicit ordering contract."""

    def __init__(self, length: int, dimension: int, order: str, device: torch.device):
        if order not in ("oldest_to_newest", "newest_to_oldest"):
            raise ValueError("order must be oldest_to_newest or newest_to_oldest")
        self.length = length
        self.dimension = dimension
        self.order = order
        self.buffer = torch.zeros((1, length, dimension), dtype=torch.float32, device=device)
        self.initialized = False

    def reset(self):
        self.buffer.zero_()
        self.initialized = False

    def push(self, frame: torch.Tensor, repeat_on_first: bool = False) -> torch.Tensor:
        if frame.ndim == 1:
            frame = frame.unsqueeze(0)
        if tuple(frame.shape) != (1, self.dimension):
            raise ValueError(f"history frame must be (1,{self.dimension}), got {tuple(frame.shape)}")
        frame = frame.to(device=self.buffer.device, dtype=self.buffer.dtype)
        if not self.initialized and repeat_on_first:
            self.buffer[:] = frame.unsqueeze(1)
        elif self.order == "oldest_to_newest":
            self.buffer = torch.cat((self.buffer[:, 1:], frame.unsqueeze(1)), dim=1)
        else:
            self.buffer = torch.cat((frame.unsqueeze(1), self.buffer[:, :-1]), dim=1)
        self.initialized = True
        return self.buffer.reshape(1, self.length * self.dimension)

    def value(self) -> torch.Tensor:
        return self.buffer.reshape(1, self.length * self.dimension)

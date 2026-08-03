import json
import os
from typing import Optional

import torch


class LoadedPolicy:
    def __init__(self, path: str, device: torch.device):
        self.path = os.path.abspath(path)
        self.device = device
        self.policy = torch.jit.load(self.path, map_location=device).eval()


def load_navigation_policy(path: str, metadata_path: str, device: torch.device) -> LoadedPolicy:
    loaded = LoadedPolicy(path, device)
    if metadata_path:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("observation_dimension") != 550 or metadata.get("action_dimension") != 3:
            raise ValueError("navigation metadata is not the SEA-Nav 550->3 contract")
    return loaded


def load_himloco_policy(path: str, device: torch.device) -> LoadedPolicy:
    return LoadedPolicy(path, device)


def infer(policy: LoadedPolicy, observation: torch.Tensor, action_dim: int) -> torch.Tensor:
    with torch.inference_mode():
        action = policy.policy(observation.to(policy.device))
    if action.ndim != 2 or action.shape[0] != observation.shape[0] or action.shape[1] != action_dim:
        raise RuntimeError(f"policy output must be [N,{action_dim}], got {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise FloatingPointError("policy output contains NaN or Inf")
    return action

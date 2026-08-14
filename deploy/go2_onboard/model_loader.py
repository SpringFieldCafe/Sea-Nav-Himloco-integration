import json
import os
import hashlib
from typing import Optional

import torch


class LoadedPolicy:
    def __init__(self, path: str, device: torch.device):
        self.path = os.path.abspath(path)
        self.device = device
        self.policy = torch.jit.load(self.path, map_location=device).eval()
        self.sha256 = _sha256(self.path)
        self.size_bytes = os.path.getsize(self.path)


def load_navigation_policy(path: str, metadata_path: str, device: torch.device) -> LoadedPolicy:
    loaded = LoadedPolicy(path, device)
    _validate_shape(loaded, 550, 3)
    if metadata_path:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("observation_dimension") != 550 or metadata.get("action_dimension") != 3:
            raise ValueError("navigation metadata is not the SEA-Nav 550->3 contract")
    return loaded


def load_himloco_policy(path: str, device: torch.device) -> LoadedPolicy:
    loaded = LoadedPolicy(path, device)
    _validate_shape(loaded, 270, 12)
    return loaded


def infer(policy: LoadedPolicy, observation: torch.Tensor, action_dim: int) -> torch.Tensor:
    with torch.inference_mode():
        action = policy.policy(observation.to(policy.device))
    if action.ndim != 2 or action.shape[0] != observation.shape[0] or action.shape[1] != action_dim:
        raise RuntimeError(f"policy output must be [N,{action_dim}], got {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise FloatingPointError("policy output contains NaN or Inf")
    return action


def _validate_shape(policy: LoadedPolicy, input_dim: int, output_dim: int) -> None:
    with torch.inference_mode():
        output = policy.policy(torch.zeros((1, input_dim), dtype=torch.float32, device=policy.device))
    if tuple(output.shape) != (1, output_dim):
        raise RuntimeError(f"policy {policy.path} must implement {input_dim}->{output_dim}, got {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise FloatingPointError(f"policy {policy.path} returned NaN or Inf during contract probe")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

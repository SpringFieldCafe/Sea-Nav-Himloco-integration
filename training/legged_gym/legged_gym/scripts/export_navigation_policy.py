"""Export the SEA-Nav navigation actor for onboard inference.

This script deliberately builds the actor with the same Go2PosRough PPO
configuration used by ``play.py`` and loads only ``model_state_dict`` from an
RSL-RL checkpoint.  It does not create an Isaac Gym simulation and does not
export the critic or optimizer state.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import isaacgym  # noqa: F401  (must be imported before torch)
import torch
import torch.nn as nn
import torch.nn.functional as F

from legged_gym.envs.go2.go2_pos_config import Go2PosRoughCfg, Go2PosRoughCfgPPO
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import OnPolicyRunner


class _ExportEnv:
    """Minimal runner environment descriptor; no simulator is constructed."""

    def __init__(self, env_cfg):
        self.cfg = env_cfg
        self.num_envs = 1
        self.num_obs = env_cfg.env.num_observations
        self.num_nav_actions = env_cfg.env.num_nav_actions
        self.num_props = env_cfg.env.num_props
        self.rays = torch.zeros((1, env_cfg.env.num_rays), dtype=torch.float32)

    def reset(self):
        return torch.zeros((1, self.num_obs)), None


class NavigationActorExport(nn.Module):
    """Critic-free TorchScript wrapper matching actor ``act_inference``."""

    def __init__(self, actor_critic):
        super().__init__()
        self.encoder = actor_critic.encoder
        self.backbone = actor_critic.backbone
        self.nav_head = actor_critic.nav_head
        self.alpha_head = actor_critic.alpha_head
        self.cbf_layer = actor_critic.cbf_layer
        self.num_obs_one_step = actor_critic.num_obs_one_step
        self.num_props = actor_critic.num_props
        self.num_rays = actor_critic.num_rays

    def forward(self, observations):
        obs_buf = observations[:, -self.num_obs_one_step:]
        props = obs_buf[:, :self.num_props]
        rays = obs_buf[:, self.num_props:self.num_props + self.num_rays]
        del props  # documents the actor contract; props remain in obs_buf.

        latent = self.encoder(observations)
        obs_cat = torch.cat((obs_buf, latent), dim=-1)
        shared_features = self.backbone(obs_cat)
        u_bar = self.nav_head(shared_features)
        alpha_raw = self.alpha_head(shared_features)
        rays_real = torch.exp2(rays)
        alpha = F.softplus(alpha_raw)
        return self.cbf_layer(u_bar, rays_real, alpha)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_runner(checkpoint):
    env_cfg = Go2PosRoughCfg()
    train_cfg = Go2PosRoughCfgPPO()
    env = _ExportEnv(env_cfg)
    runner = OnPolicyRunner(
        env,
        class_to_dict(train_cfg),
        log_dir=None,
        args=SimpleNamespace(),
        device="cpu",
    )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_data, dict) or "model_state_dict" not in checkpoint_data:
        raise ValueError("Expected an RSL-RL checkpoint with model_state_dict")
    runner.alg.actor_critic.load_state_dict(checkpoint_data["model_state_dict"])
    runner.alg.actor_critic.eval()
    return runner, checkpoint_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument(
        "--output",
        default="artifacts/go2_onboard/sea_nav_policy_2000.pt",
    )
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    output = os.path.abspath(args.output)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    runner, checkpoint_data = _build_runner(checkpoint)
    actor = NavigationActorExport(runner.alg.actor_critic).eval().cpu()
    input_dim = runner.env.num_obs
    action_dim = runner.env.num_nav_actions

    os.makedirs(os.path.dirname(output), exist_ok=True)
    scripted = torch.jit.script(actor)
    scripted.save(output)

    scripted = torch.jit.load(output, map_location="cpu").eval()
    dummy = torch.zeros((1, input_dim), dtype=torch.float32)
    with torch.inference_mode():
        result = scripted(dummy)
    if tuple(result.shape) != (1, action_dim):
        raise RuntimeError(f"Unexpected export output shape: {tuple(result.shape)}")
    if result.dtype != torch.float32:
        raise RuntimeError(f"Unexpected export output dtype: {result.dtype}")
    if not torch.isfinite(result).all():
        raise FloatingPointError("Exported policy returned NaN or Inf")

    metadata = {
        "original_checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "exported_model": output,
        "exported_model_sha256": _sha256(output),
        "observation_dimension": input_dim,
        "action_dimension": action_dim,
        "observation_dtype": "float32",
        "action_dtype": "float32",
        "observation_history_length": 10,
        "observation_history_order": "oldest_to_newest",
        "observation_fields_per_frame": [
            {"name": "projected_gravity", "dim": 3, "scale": 1.0},
            {"name": "navigation_command_scaled", "dim": 3, "scale": [2.0, 2.0, 0.25]},
            {"name": "base_linear_velocity", "dim": 3, "scale": 1.0},
            {"name": "base_angular_velocity", "dim": 3, "scale": 1.0},
            {"name": "ray_distance_log2", "dim": 41, "clip": [0.1, 5.0]},
            {"name": "relative_goal_xy", "dim": 2, "scale": 1.0},
        ],
        "action_semantics": ["vx", "vy", "wz"],
        "action_scale": [1.0, 1.0, 1.0],
        "runtime_navigation_clip": [-3.0, 3.0],
        "himloco_command_bounds": {
            "vx": [-1.0, 1.0],
            "vy": [-1.0, 1.0],
            "wz": [-2.0, 2.0],
        },
        "checkpoint_iteration": checkpoint_data.get("iter"),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "torchscript": True,
    }
    metadata_path = os.path.splitext(output)[0] + ".json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"Exported navigation actor: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Checkpoint iteration: {checkpoint_data.get('iter')}")
    print(f"CPU inference: {tuple(result.shape)} {result.dtype} finite={bool(torch.isfinite(result).all())}")
    print(f"Output SHA256: {_sha256(output)}")


if __name__ == "__main__":
    main()

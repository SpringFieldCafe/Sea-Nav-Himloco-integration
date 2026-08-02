"""Locomotion policy backends used by the navigation environment.

The backend owns policy-specific observation construction and inference.  The
environment remains responsible for navigation filtering, simulation stepping,
and episode bookkeeping.
"""

import os

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR


class LocomotionBackend:
    """Interface shared by selectable locomotion controllers."""

    name = "base"

    def __init__(self, env):
        self.env = env

    def compute_actions(self, nav_actions):
        raise NotImplementedError

    def reset(self, env_ids):
        """Reset policy state for the selected environments."""

    def control_params(self):
        """Return policy-specific low-level control parameters."""
        raise NotImplementedError

    def uses_identity_action_order(self):
        """Whether policy actions already use the simulation DOF order."""
        return False


class SLRBackend(LocomotionBackend):
    """Original SEA-Nav SLR controller, extracted without changing its math."""

    name = "slr"

    def __init__(self, env):
        super().__init__(env)
        self.body = torch.jit.load(
            os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "ctrl_model", "body_latest.jit")
        )
        self.encoder_vel = torch.jit.load(
            os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "ctrl_model", "encoder_vel.jit")
        )
        self.encoder_latent = torch.jit.load(
            os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "ctrl_model", "encoder_latent.jit")
        )

    def compute_actions(self, nav_actions):
        env = self.env
        env.slr_commands = nav_actions

        scale_lin_vel = env.cfg.loco.normalization.obs_scales.lin_vel
        scale_ang_vel = env.cfg.loco.normalization.obs_scales.ang_vel
        scale_dof_pos = env.cfg.loco.normalization.obs_scales.dof_pos
        scale_dof_vel = env.cfg.loco.normalization.obs_scales.dof_vel

        env.slr_commands_scale = torch.tensor(
            [scale_lin_vel, scale_lin_vel, scale_ang_vel],
            device=env.device,
            requires_grad=False,
        )
        env.slr_obs_buf = torch.cat(
            (
                env.base_ang_vel * scale_ang_vel,
                env.projected_gravity,
                env.slr_commands[:, :3] * env.slr_commands_scale,
                env.reindex((env.dof_pos - env.default_dof_pos) * scale_dof_pos),
                env.reindex(env.dof_vel * scale_dof_vel),
                env.actions_orig,
            ),
            dim=-1,
        )

        noise_scales = env.cfg.noise.noise_scales
        noise_vec = torch.cat(
            (
                torch.ones(3) * noise_scales.ang_vel,
                torch.ones(3) * noise_scales.gravity,
                torch.zeros(3),
                torch.ones(12) * noise_scales.dof_pos * env.obs_scales.dof_pos,
                torch.ones(12) * noise_scales.dof_vel * env.obs_scales.dof_vel,
                torch.zeros(env.num_actions),
            ),
            dim=0,
        )

        if env.cfg.noise.add_noise:
            env.slr_obs_buf += (2 * torch.rand_like(env.slr_obs_buf) - 1) * 0.5 * noise_vec.to(env.device)

        env.slr_obs_hist = torch.where(
            (env.episode_length_buf <= 1)[:, None, None],
            torch.stack([env.slr_obs_buf] * env.cfg.env.his_len, dim=1),
            torch.cat(
                [env.slr_obs_hist[:, 1:], env.slr_obs_buf.unsqueeze(1)],
                dim=1,
            ),
        )
        prop = env.slr_obs_buf
        ang_vel = env.base_ang_vel[:, 2:] * scale_ang_vel
        env.base_lin_vel_pred = self.encoder_vel(env.slr_obs_hist.view(env.num_envs, -1))
        latent = self.encoder_latent(env.slr_obs_hist.view(env.num_envs, -1))
        actor_obs = torch.cat((env.base_lin_vel_pred, prop, ang_vel, latent), dim=-1)
        return self.body(actor_obs)

    def control_params(self):
        return {
            "action_scale": 0.25,
            "default_dof_pos": self.env.default_dof_pos,
            "p_gains": self.env.p_gains,
            "d_gains": self.env.d_gains,
        }


def make_locomotion_backend(env, backend_name):
    if backend_name == "slr":
        return SLRBackend(env)
    raise ValueError(f"Unsupported locomotion backend: {backend_name}")

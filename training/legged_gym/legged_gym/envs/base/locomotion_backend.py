"""Locomotion policy backends used by the navigation environment.

The backend owns policy-specific observation construction and inference.  The
environment remains responsible for navigation filtering, simulation stepping,
and episode bookkeeping.
"""

import os
from typing import List

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

    def command_bounds(self):
        return None

    def clip_actions(self, actions):
        clip = self.env.cfg.normalization.clip_actions
        return torch.clip(actions, -clip, clip)

    def record_action(self, actions):
        """Record the action after environment-level clipping."""

    def record_commands(self, before_clip, after_clip):
        """Record command clipping diagnostics without printing every step."""

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


class HIMLocoBackend(LocomotionBackend):
    """HIMLoco-Go2 TorchScript policy and observation adapter."""

    name = "himloco"
    POLICY_JOINT_NAMES = (
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    )
    ACTION_DIM = 12

    def __init__(self, env):
        super().__init__(env)
        self.contract = env.cfg.locomotion.himloco
        self.num_one_step_obs = self.contract.num_one_step_obs
        self.history_length = self.contract.history_length
        self.input_dim = self.num_one_step_obs * self.history_length
        if self.num_one_step_obs != 45 or self.history_length != 6:
            raise ValueError("HIMLoco contract must be 45 observations over 6 history frames")
        policy_path = getattr(env.cfg.locomotion, "himloco_policy", None)
        if not policy_path:
            raise ValueError(
                "HIMLoco backend requires --himloco_policy PATH; "
                "no default model path is configured."
            )
        if not os.path.isfile(policy_path):
            raise FileNotFoundError(f"HIMLoco policy does not exist: {policy_path}")

        self.policy_path = os.path.abspath(policy_path)
        self.policy = torch.jit.load(self.policy_path, map_location=env.device)
        self.policy = self.policy.to(env.device).eval()
        self.history = torch.zeros(
            env.num_envs,
            self.history_length,
            self.num_one_step_obs,
            device=env.device,
            dtype=torch.float32,
        )
        self.previous_action = torch.zeros(
            env.num_envs, self.ACTION_DIM, device=env.device, dtype=torch.float32
        )
        self._reported_anomaly = False
        self.command_samples = 0
        self.command_clipped_samples = 0
        self.command_min = torch.full((3,), float("inf"), device=env.device)
        self.command_max = torch.full((3,), float("-inf"), device=env.device)
        self.last_observation = None
        self.last_one_step_observation = None
        self.last_policy_action = None
        self.last_command = None
        self.last_raw_angular_velocity = None
        self.last_raw_gravity = None
        self.last_raw_dof_position = None
        self.last_raw_dof_velocity = None
        self.last_scaled_command = None
        self.last_scaled_angular_velocity = None
        self.last_scaled_dof_position = None
        self.last_scaled_dof_velocity = None

        self.policy_to_sim = self._build_joint_mapping(env.dof_names)
        self.policy_to_sim_device = self.policy_to_sim.to(env.device)
        self.sim_to_policy = self._build_inverse_mapping(self.policy_to_sim)
        self.sim_to_policy_device = self.sim_to_policy.to(env.device)
        self.default_dof_pos = self._build_default_positions(env)
        self.p_gains = torch.full(
            (self.ACTION_DIM,), self.contract.p_gain, device=env.device, dtype=torch.float32
        )
        self.d_gains = torch.full(
            (self.ACTION_DIM,), self.contract.d_gain, device=env.device, dtype=torch.float32
        )

        with torch.inference_mode():
            probe = torch.zeros(1, self.input_dim, device=env.device)
            output = self.policy(probe)
        self._assert_policy_output(output, 1)
        self._print_diagnostics(env)

    @classmethod
    def _build_joint_mapping(cls, dof_names: List[str]) -> torch.Tensor:
        if len(dof_names) != cls.ACTION_DIM or len(set(dof_names)) != cls.ACTION_DIM:
            raise ValueError(f"Expected 12 unique Isaac Gym DOF names, got {dof_names}")
        missing = [name for name in cls.POLICY_JOINT_NAMES if name not in dof_names]
        if missing:
            raise ValueError(f"HIMLoco joint names missing from Isaac Gym DOFs: {missing}")
        mapping = torch.tensor(
            [dof_names.index(name) for name in cls.POLICY_JOINT_NAMES], dtype=torch.long
        )
        if len(set(mapping.tolist())) != cls.ACTION_DIM or sorted(mapping.tolist()) != list(range(cls.ACTION_DIM)):
            raise ValueError(f"HIMLoco policy_to_sim is not a permutation: {mapping.tolist()}")
        return mapping

    @staticmethod
    def _build_inverse_mapping(policy_to_sim):
        sim_to_policy = torch.empty_like(policy_to_sim)
        sim_to_policy[policy_to_sim] = torch.arange(policy_to_sim.numel(), dtype=torch.long)
        if not torch.equal(sim_to_policy[policy_to_sim], torch.arange(policy_to_sim.numel())):
            raise ValueError("HIMLoco joint mapping round-trip check failed")
        return sim_to_policy

    def _build_default_positions(self, env):
        policy_defaults = torch.tensor(
            [self.contract.default_joint_angles[name] for name in self.POLICY_JOINT_NAMES],
            device=env.device,
            dtype=torch.float32,
        )
        defaults = torch.zeros(self.ACTION_DIM, device=env.device, dtype=torch.float32)
        defaults[self.policy_to_sim_device] = policy_defaults
        return defaults.unsqueeze(0)

    def _print_diagnostics(self, env):
        sim_frequency = 1.0 / float(env.dt)
        command_ranges = self.command_bounds()
        print("[HIMLoco] policy path:", self.policy_path)
        print(f"[HIMLoco] input dimension: {self.input_dim}, history length: {self.history_length}, output dimension: 12")
        print("[HIMLoco] joint order:", ", ".join(self.POLICY_JOINT_NAMES))
        print(f"[HIMLoco] action scale: {self.contract.action_scale}, Kp: {self.contract.p_gain}, Kd: {self.contract.d_gain}")
        print("[HIMLoco] policy control frequency: 50.0 Hz")
        print(f"[HIMLoco] simulation control frequency: {sim_frequency:.3f} Hz")
        print("[HIMLoco] command scales:", list(self.contract.command_scale))
        print(f"[HIMLoco] policy action clip: {self.contract.action_clip}")
        print(
            "[HIMLoco] command bounds min/max:",
            command_ranges[0].tolist(), command_ranges[1].tolist(),
        )

    def _assert_policy_output(self, output, batch_size):
        if output.ndim != 2 or tuple(output.shape) != (batch_size, self.ACTION_DIM):
            raise RuntimeError(
                f"HIMLoco policy output must be ({batch_size}, 12), got {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            self._report_anomaly("action", output)
            raise FloatingPointError("HIMLoco policy output contains NaN or Inf")

    def _report_anomaly(self, name, value):
        if self._reported_anomaly:
            return
        self._reported_anomaly = True
        finite = torch.isfinite(value)
        print(
            f"[HIMLoco] invalid {name}: shape={tuple(value.shape)}, "
            f"finite={bool(finite.all())}, min={value.nan_to_num().min().item()}, "
            f"max={value.nan_to_num().max().item()}"
        )

    def compute_actions(self, nav_actions):
        env = self.env
        env.slr_commands = nav_actions
        command = nav_actions[:, :3]
        joint_pos = env.dof_pos[:, self.policy_to_sim_device] - self.default_dof_pos[:, self.policy_to_sim_device]
        joint_vel = env.dof_vel[:, self.policy_to_sim_device]
        one_step = torch.cat(
            (
                command * torch.tensor(self.contract.command_scale, device=env.device),
                env.base_ang_vel[:, :3] * self.contract.angular_velocity_scale,
                env.projected_gravity,
                joint_pos * self.contract.dof_position_scale,
                joint_vel * self.contract.dof_velocity_scale,
                self.previous_action,
            ),
            dim=-1,
        )
        self.last_raw_angular_velocity = env.base_ang_vel[:, :3].detach().clone()
        self.last_raw_gravity = env.projected_gravity.detach().clone()
        self.last_raw_dof_position = joint_pos.detach().clone()
        self.last_raw_dof_velocity = joint_vel.detach().clone()
        self.last_scaled_command = (command * torch.tensor(self.contract.command_scale, device=env.device)).detach().clone()
        self.last_scaled_angular_velocity = (env.base_ang_vel[:, :3] * self.contract.angular_velocity_scale).detach().clone()
        self.last_scaled_dof_position = (joint_pos * self.contract.dof_position_scale).detach().clone()
        self.last_scaled_dof_velocity = (joint_vel * self.contract.dof_velocity_scale).detach().clone()
        if one_step.shape[-1] != self.num_one_step_obs:
            raise RuntimeError(f"HIMLoco observation must be 45D, got {one_step.shape[-1]}")
        if not torch.isfinite(one_step).all():
            self._report_anomaly("observation", one_step)
            raise FloatingPointError("HIMLoco observation contains NaN or Inf")
        self.history = torch.cat((one_step.unsqueeze(1), self.history[:, :-1]), dim=1)
        policy_input = self.history.reshape(env.num_envs, self.input_dim)
        if policy_input.shape[-1] != self.input_dim:
            raise RuntimeError(f"HIMLoco input must be 270D, got {policy_input.shape[-1]}")
        if not torch.isfinite(policy_input).all():
            self._report_anomaly("input", policy_input)
            raise FloatingPointError("HIMLoco input contains NaN or Inf")

        with torch.inference_mode():
            policy_action = self.policy(policy_input)
        self._assert_policy_output(policy_action, env.num_envs)
        self.last_observation = policy_input.detach().clone()
        self.last_one_step_observation = one_step.detach().clone()
        self.last_policy_action = policy_action.detach().clone()
        self.last_command = command.detach().clone()
        sim_action = torch.zeros_like(policy_action)
        sim_action[:, self.policy_to_sim_device] = policy_action
        return sim_action

    def reset(self, env_ids):
        self.previous_action[env_ids] = 0.0

    def command_bounds(self):
        bounds = torch.tensor(
            [self.contract.command_ranges[name] for name in ("vx", "vy", "wz")],
            device=self.env.device,
            dtype=torch.float32,
        )
        return bounds[:, 0], bounds[:, 1]

    def clip_actions(self, actions):
        return torch.clip(actions, -self.contract.action_clip, self.contract.action_clip)

    def record_action(self, actions):
        self.previous_action = actions[:, self.policy_to_sim_device].detach()

    def record_commands(self, before_clip, after_clip):
        self.command_samples += before_clip.shape[0]
        self.command_clipped_samples += int(torch.any(before_clip != after_clip, dim=1).sum().item())
        self.command_min = torch.minimum(self.command_min, after_clip.detach().min(dim=0).values)
        self.command_max = torch.maximum(self.command_max, after_clip.detach().max(dim=0).values)

    def command_stats(self):
        ratio = self.command_clipped_samples / max(self.command_samples, 1)
        return {
            "samples": self.command_samples,
            "clipped_samples": self.command_clipped_samples,
            "clip_ratio": ratio,
            "min": self.command_min.detach().cpu().tolist(),
            "max": self.command_max.detach().cpu().tolist(),
        }

    def control_params(self):
        return {
            "action_scale": self.contract.action_scale,
            "default_dof_pos": self.default_dof_pos,
            "p_gains": self.p_gains,
            "d_gains": self.d_gains,
        }

    def uses_identity_action_order(self):
        return True


def make_locomotion_backend(env, backend_name):
    if backend_name == "slr":
        return SLRBackend(env)
    if backend_name == "himloco":
        return HIMLocoBackend(env)
    raise ValueError(f"Unsupported locomotion backend: {backend_name}")

"""Regression checks for the SLR extraction.

The reference below is deliberately independent of ``SLRBackend``.  It is a
small transcription of the pre-refactor implementation from ``f0275b3^`` so
the test does not merely compare a function with itself.
"""

from types import SimpleNamespace
from unittest.mock import patch

# Isaac Gym must be imported before torch in the sea_nav environment.
import isaacgym  # noqa: F401
import torch


REINDEX = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


class _Encoder(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, value):
        return value[:, : self.width]


class _Body(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, value):
        self.last_input = value.detach().clone()
        return value[:, :12]


def _make_env():
    scales = SimpleNamespace(lin_vel=2.0, ang_vel=0.25, dof_pos=1.0, dof_vel=0.05)
    noise_scales = SimpleNamespace(ang_vel=0.0, gravity=0.0, dof_pos=0.0, dof_vel=0.0)
    cfg = SimpleNamespace(
        loco=SimpleNamespace(normalization=SimpleNamespace(obs_scales=scales)),
        normalization=SimpleNamespace(clip_actions=100.0),
        noise=SimpleNamespace(add_noise=False, noise_scales=noise_scales),
        env=SimpleNamespace(his_len=3),
    )
    env = SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=2,
        num_actions=12,
        cfg=cfg,
        obs_scales=scales,
        base_ang_vel=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0], [0.1, 0.2, -0.9]]),
        slr_commands=None,
        dof_pos=torch.arange(24, dtype=torch.float32).reshape(2, 12) / 10,
        default_dof_pos=torch.zeros(1, 12),
        dof_vel=torch.arange(24, dtype=torch.float32).reshape(2, 12) / 20,
        actions_orig=torch.arange(24, dtype=torch.float32).reshape(2, 12) / 30,
        slr_obs_hist=torch.zeros(2, 3, 45),
        slr_obs_buf=None,
        slr_commands_scale=None,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        base_lin_vel_pred=None,
        dof_names=[
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ],
    )
    env.reindex = lambda value: value[:, REINDEX]
    return env


def _reference_slr_compute(env, nav_actions, body, encoder_vel, encoder_latent):
    """Independent reference copied from the pre-backend _compute_actions."""
    env.slr_commands = nav_actions
    scales = env.cfg.loco.normalization.obs_scales
    env.slr_commands_scale = torch.tensor([scales.lin_vel, scales.lin_vel, scales.ang_vel])
    env.slr_obs_buf = torch.cat(
        (
            env.base_ang_vel * scales.ang_vel,
            env.projected_gravity,
            env.slr_commands[:, :3] * env.slr_commands_scale,
            env.reindex((env.dof_pos - env.default_dof_pos) * scales.dof_pos),
            env.reindex(env.dof_vel * scales.dof_vel),
            env.actions_orig,
        ),
        dim=-1,
    )
    env.slr_obs_hist = torch.where(
        (env.episode_length_buf <= 1)[:, None, None],
        torch.stack([env.slr_obs_buf] * env.cfg.env.his_len, dim=1),
        torch.cat([env.slr_obs_hist[:, 1:], env.slr_obs_buf.unsqueeze(1)], dim=1),
    )
    prop = env.slr_obs_buf
    ang_vel = env.base_ang_vel[:, 2:] * scales.ang_vel
    env.base_lin_vel_pred = encoder_vel(env.slr_obs_hist.view(env.num_envs, -1))
    latent = encoder_latent(env.slr_obs_hist.view(env.num_envs, -1))
    return body(torch.cat((env.base_lin_vel_pred, prop, ang_vel, latent), dim=-1))


def test_slr_backend_matches_pre_refactor_reference():
    from legged_gym.envs.base.locomotion_backend import SLRBackend

    env = _make_env()
    body = _Body()
    encoder_vel = _Encoder(3)
    encoder_latent = _Encoder(4)

    def fake_load(path, *args, **kwargs):
        if "body" in path:
            return body
        if "encoder_vel" in path:
            return encoder_vel
        return encoder_latent

    with patch("torch.jit.load", side_effect=fake_load):
        backend = SLRBackend(env)

    commands = torch.tensor([[0.2, -0.3, 0.4], [1.1, 0.5, -0.7]])
    expected = _reference_slr_compute(env, commands, body, encoder_vel, encoder_latent)
    actual = backend.compute_actions(commands)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(env.slr_obs_hist, env.slr_obs_hist)

    # Compare a rolling-history step independently, after the reset branch ends.
    env.episode_length_buf[:] = 2
    env.base_ang_vel += 0.3
    env.dof_vel += 0.2
    env.actions_orig -= 0.1
    reference_history = env.slr_obs_hist.clone()
    expected = _reference_slr_compute(env, commands * 0.5, body, encoder_vel, encoder_latent)
    reference_history = env.slr_obs_hist.clone()

    env2 = _make_env()
    with patch("torch.jit.load", side_effect=fake_load):
        backend2 = SLRBackend(env2)
    backend2.compute_actions(commands)
    env2.episode_length_buf[:] = 2
    env2.base_ang_vel += 0.3
    env2.dof_vel += 0.2
    env2.actions_orig -= 0.1
    actual = backend2.compute_actions(commands * 0.5)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(env2.slr_obs_hist, reference_history, rtol=1e-6, atol=1e-7)


def test_slr_action_reindex_clip_and_torque_contract():
    from legged_gym.envs.base.legged_robot_pos import LeggedRobotPos
    from legged_gym.envs.base.locomotion_backend import SLRBackend

    env = _make_env()
    with patch("torch.jit.load", side_effect=lambda path, *a, **k: _Body() if "body" in path else _Encoder(3 if "vel" in path else 4)):
        backend = SLRBackend(env)
    env.locomotion_backend = backend
    env.dof_pos = torch.linspace(-0.4, 0.4, 24).reshape(2, 12)
    env.dof_vel = torch.linspace(-0.2, 0.2, 24).reshape(2, 12)
    env.p_gains = torch.full((12,), 30.0)
    env.d_gains = torch.full((12,), 0.75)
    env.default_dof_pos = torch.linspace(-0.3, 0.3, 12).reshape(1, 12)
    env.torque_limits = torch.full((12,), 7.5)

    raw = torch.tensor([[150.0, -150.0] + [0.0] * 10, [1.0] * 12])
    clipped = backend.clip_actions(raw)
    torch.testing.assert_close(clipped, torch.clip(raw, -100.0, 100.0))
    reindexed = LeggedRobotPos._reindex_actions_for_sim(env, raw)
    torch.testing.assert_close(reindexed, raw[:, REINDEX])

    expected = torch.clip(
        env.p_gains * (clipped * 0.25 + env.default_dof_pos - env.dof_pos)
        - env.d_gains * env.dof_vel,
        -env.torque_limits,
        env.torque_limits,
    )
    actual = LeggedRobotPos._compute_torques(env, clipped)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_default_backend_is_slr_and_does_not_require_him_policy():
    from legged_gym.envs.go2.go2_pos_config import Go2PosRoughCfg

    cfg = Go2PosRoughCfg()
    assert cfg.locomotion.backend == "slr"
    assert cfg.locomotion.himloco_policy is None


if __name__ == "__main__":
    test_slr_backend_matches_pre_refactor_reference()
    test_slr_action_reindex_clip_and_torque_contract()
    test_default_backend_is_slr_and_does_not_require_him_policy()
    print("SLR_BACKEND_REGRESSION_OK")

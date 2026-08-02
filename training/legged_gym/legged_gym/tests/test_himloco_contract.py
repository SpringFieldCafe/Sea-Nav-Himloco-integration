"""Contract and timing checks for the 270-D stacked HIMLoco actor."""

import os
from types import SimpleNamespace

import isaacgym  # noqa: F401
import torch


POLICY_ENV_VAR = "HIMLOCO_POLICY"
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


def _make_env(policy_path):
    from legged_gym.envs.go2.go2_pos_config import Go2PosRoughCfg

    cfg = Go2PosRoughCfg()
    cfg.locomotion.himloco_policy = policy_path
    return SimpleNamespace(
        cfg=cfg,
        device=torch.device("cpu"),
        num_envs=2,
        num_actions=12,
        dof_names=JOINT_NAMES,
        dt=0.02,
        base_ang_vel=torch.zeros(2, 3),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        dof_pos=torch.zeros(2, 12),
        dof_vel=torch.zeros(2, 12),
        default_dof_pos=torch.zeros(1, 12),
        actions_orig=torch.zeros(2, 12),
        slr_commands=None,
    )


def test_himloco_contract_history_mapping_reset_and_command_clip():
    policy_path = os.environ.get(POLICY_ENV_VAR)
    if not policy_path:
        return

    from legged_gym.envs.base.locomotion_backend import HIMLocoBackend

    env = _make_env(policy_path)
    backend = HIMLocoBackend(env)
    assert backend.num_one_step_obs == 45
    assert backend.history_length == 6
    assert backend.input_dim == 270
    assert torch.equal(backend.policy_to_sim[backend.sim_to_policy], torch.arange(12))
    assert torch.equal(backend.sim_to_policy[backend.policy_to_sim], torch.arange(12))

    commands = torch.tensor([[0.2, -0.1, 0.3], [0.5, 0.0, -0.4]])
    backend.compute_actions(commands)
    first_history = backend.history.clone()
    assert torch.count_nonzero(first_history[:, 0]).item() > 0
    assert torch.count_nonzero(first_history[:, 1:]).item() == 0

    env.base_ang_vel[:] = 0.25
    env.dof_pos[:] = 0.1
    env.dof_vel[:] = -0.2
    next_commands = commands * 0.5
    backend.compute_actions(next_commands)
    assert torch.count_nonzero(backend.history[:, 0] - first_history[:, 0]).item() > 0
    torch.testing.assert_close(backend.history[:, 1], first_history[:, 0])

    raw_action = torch.full((2, 12), 150.0)
    clipped_action = backend.clip_actions(raw_action)
    assert torch.equal(clipped_action, torch.full((2, 12), 100.0))
    backend.record_action(clipped_action)
    torch.testing.assert_close(backend.previous_action, clipped_action[:, backend.policy_to_sim])

    history_before_reset = backend.history.clone()
    backend.reset(torch.tensor([0], dtype=torch.long))
    torch.testing.assert_close(backend.history, history_before_reset)
    assert torch.count_nonzero(backend.previous_action[0]).item() == 0
    assert torch.count_nonzero(backend.previous_action[1]).item() == 12

    lower, upper = backend.command_bounds()
    out_of_range = torch.tensor([[2.0, -2.0, 3.0], [0.0, 0.0, 0.0]])
    clipped_commands = torch.maximum(torch.minimum(out_of_range, upper), lower)
    backend.record_commands(out_of_range, clipped_commands)
    stats = backend.command_stats()
    assert stats["samples"] == 2
    assert stats["clipped_samples"] == 1
    assert stats["clip_ratio"] == 0.5
    torch.testing.assert_close(clipped_commands[0], torch.tensor([1.0, -1.0, 2.0]))


if __name__ == "__main__":
    test_himloco_contract_history_mapping_reset_and_command_clip()
    print("HIMLOCO_CONTRACT_OK")

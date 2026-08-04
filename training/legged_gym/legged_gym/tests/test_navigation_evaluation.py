"""Static contract checks for the Isaac Gym hard-room evaluation path."""

from pathlib import Path
import shutil

import pytest


def test_hard_room_navigation_contract():
    pytest.importorskip("isaacgym")
    if shutil.which("ninja") is None:
        pytest.skip("Isaac Gym gymtorch extension tests require ninja")
    from legged_gym.envs.go2.go2_pos_config import Go2PosRoughCfg

    cfg = Go2PosRoughCfg
    assert cfg.terrain.terrain_types == ["hard_room"]
    assert cfg.env.num_rays == 41
    assert cfg.env.num_goal_obs == 2
    assert cfg.env.num_observations == 550
    assert cfg.locomotion.himloco.num_one_step_obs == 45
    assert cfg.locomotion.himloco.history_length == 6
    assert cfg.locomotion.himloco.action_scale == 0.25
    assert cfg.locomotion.himloco.command_ranges == {
        "vx": [-1.0, 1.0],
        "vy": [-1.0, 1.0],
        "wz": [-2.0, 2.0],
    }


def test_play_exposes_seeded_evaluation_options():
    source = Path(__file__).parents[1] / "scripts" / "play.py"
    text = source.read_text(encoding="utf-8")
    for option in ("--eval_log", "--eval_episodes", "--eval_steps", "--eval_log_interval"):
        assert option in text

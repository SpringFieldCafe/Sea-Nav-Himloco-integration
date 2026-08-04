import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deploy.go2_onboard.himloco_observation import HIMLocoObservation
from deploy.go2_onboard.joint_mapping import make_motor_to_policy, make_policy_to_motor
from .state import UnifiedState
from .supervisor import TerrainState, TerrainSupervisor


def _state(terrain="flat"):
    return UnifiedState(np.array([0, 0, -1], dtype=np.float32), np.zeros(3, dtype=np.float32),
                        np.zeros(3, dtype=np.float32), np.zeros(12, dtype=np.float32),
                        np.zeros(12, dtype=np.float32), np.zeros(2, dtype=np.float32),
                        np.ones(41, dtype=np.float32), np.array([1, 0], dtype=np.float32),
                        terrain_hint=terrain)


def test_himloco_observation_contract():
    obs = HIMLocoObservation(torch.device("cpu"))
    frame = obs.build(torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]),
                      torch.zeros(1, 12), torch.zeros(1, 12))
    assert tuple(frame.shape) == (1, 270)
    obs.record_action(torch.full((1, 12), 150.0))
    assert torch.all(obs.previous_action == 100)


def test_joint_mapping_round_trip():
    assert make_motor_to_policy()[make_policy_to_motor()[0]] == 0


def test_supervisor_limits_stairs_and_stops_on_fall():
    supervisor = TerrainSupervisor()
    command, state = supervisor.update(_state("stairs_up"), np.array([1., 1., 2.]))
    assert state == TerrainState.STAIRS_UP
    assert np.all(np.abs(command) <= np.array([.28, .12, .35]))
    command, state = supervisor.update(_state("stairs_up"), np.array([-1., 0., 0.]))
    assert state == TerrainState.STAIRS_UP
    assert command[0] > 0.0
    fallen = _state("flat")
    fallen.fallen = True
    command, state = supervisor.update(fallen, np.ones(3))
    assert state == TerrainState.EMERGENCY_STOP
    assert np.all(command == 0)

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from deploy.go2_onboard.himloco_fixed_control import (
    ACTION_SCALE,
    COMMAND_SCALE,
    DEFAULT_ANGLES,
    EXPECTED_INPUT_DIM,
    EXPECTED_OUTPUT_DIM,
    EXPECTED_SHA256,
    FixedHIMLocoController,
    ARM_WAIT_TIMEOUT,
    KD,
    KP,
    POLICY_TO_MOTOR,
    LowStateWatchdog,
    RuntimeState,
    SafetyError,
    build_observation,
    build_target_q,
    projected_gravity_from_wxyz,
    sha256_file,
    validate_fixed_command,
)
from deploy.go2_onboard.himloco_observation import HIMLocoObservation


def test_current_1460_hash_and_shape():
    path = Path("models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt")
    assert sha256_file(str(path)) == EXPECTED_SHA256
    policy = torch.jit.load(str(path), map_location="cpu").eval()
    with torch.inference_mode():
        output = policy(torch.zeros((1, EXPECTED_INPUT_DIM)))
    assert tuple(output.shape) == (1, EXPECTED_OUTPUT_DIM)


def test_fixed_observation_is_270_and_newest_first():
    him = HIMLocoObservation(torch.device("cpu"))
    observation = build_observation(
        him, [0.1, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0],
        DEFAULT_ANGLES, [0.0] * 12,
    )
    assert tuple(observation.shape) == (1, 270)
    torch.testing.assert_close(observation[0, :3], torch.tensor([0.2, 0.0, 0.025]))
    assert torch.count_nonzero(observation[0, 45:]) == 0


def test_joint_mapping_and_target_contract():
    assert POLICY_TO_MOTOR == (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
    action = np.arange(12, dtype=np.float32)
    np.testing.assert_allclose(build_target_q(action), DEFAULT_ANGLES + ACTION_SCALE * action)
    assert KP == 20.0
    assert KD == 0.5
    np.testing.assert_allclose(COMMAND_SCALE, [2.0, 2.0, 0.25])


@pytest.mark.parametrize("command", [(0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.0, 0.0, 0.15), (0.0, 0.0, -0.15)])
def test_fixed_command_whitelist(command):
    assert validate_fixed_command(*command).as_array().shape == (3,)


@pytest.mark.parametrize("command", [(-0.01, 0.0, 0.0), (0.16, 0.0, 0.0), (0.0, 0.01, 0.0), (0.1, 0.0, 0.1)])
def test_fixed_command_whitelist_rejects_unsafe_modes(command):
    with pytest.raises(SafetyError):
        validate_fixed_command(*command)


def test_watchdog_and_imu_finite_contract():
    watchdog = LowStateWatchdog(0.10)
    assert watchdog.fresh(10.0, 10.09)
    assert not watchdog.fresh(10.0, 10.11)
    np.testing.assert_allclose(projected_gravity_from_wxyz([1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, -1.0])


def test_shadow_remains_read_only():
    for name in ("shadow_worker.py", "runtime.py", "ros_state_reader.py"):
        source = Path("deploy/go2_onboard") / name
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported = {node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import)}
        imported_from = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert "unitree_go.msg.LowCmd" not in imported_from
        assert "unitree_sdk2py" not in " ".join(sorted(imported | {x or "" for x in imported_from}))
    worker = Path("deploy/go2_onboard/shadow_worker.py").read_text(encoding="utf-8")
    assert "create_publisher" not in worker
    assert ".publish(" not in worker
    assert '"lowcmd_sent": False' in worker


def test_stop_transitions_to_exit_without_transport():
    controller = object.__new__(FixedHIMLocoController)
    controller.state = RuntimeState.ACTIVE
    controller.sent_any_command = False
    called = []
    controller._send_damping = lambda: called.append(True)

    controller.stop()

    assert called == [True]
    assert controller.state == RuntimeState.EXIT


def test_lowstate_subscriber_is_retained_by_controller():
    source = Path("deploy/go2_onboard/himloco_fixed_control.py").read_text(encoding="utf-8")
    assert "self.lowstate_subscriber = ChannelSubscriber" in source
    assert "self.lowstate_subscriber.Init(self._low_state_callback, 10)" in source
    assert ARM_WAIT_TIMEOUT == 5.0
    assert "no fresh LowState received within" in source


def test_wait_for_arm_keeps_waiting_after_first_fresh_lowstate():
    source = Path("deploy/go2_onboard/himloco_fixed_control.py").read_text(encoding="utf-8")
    wait_block = source.split("        wait_deadline =", 1)[1].split("        next_tick =", 1)[0]
    assert "if not self.watchdog.fresh" in wait_block
    assert "continue" in wait_block
    assert "if key_pressed(snapshot.remote_keys, self.ARM_KEY)" in wait_block

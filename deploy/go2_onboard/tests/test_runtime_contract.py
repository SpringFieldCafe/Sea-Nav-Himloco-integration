import math
import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from deploy.go2_onboard.command_bridge import ReadOnlyCommandBridge
from deploy.go2_onboard.goal import Goal2D, GoalManager
from deploy.go2_onboard.himloco_observation import HIMLocoObservation
from deploy.go2_onboard.joint_mapping import make_motor_to_policy, make_policy_to_motor
from deploy.go2_onboard.lidar_ray_adapter import LidarRayAdapter
from deploy.go2_onboard.model_loader import load_himloco_policy, load_navigation_policy
from deploy.go2_onboard.navigation_observation import NavigationObservation
from deploy.go2_onboard.diagnostics import _ros_value
from deploy.go2_onboard.ros_state_reader import Latest
from deploy.go2_onboard.runtime import _health_for_log
from deploy.go2_onboard.safety_supervisor import RuntimeState, SafetySupervisor, _is_finite


def test_goal_transform_global_to_body_and_body_goal():
    manager = GoalManager()
    manager.update(Goal2D("odom", 2.0, 1.0, 10.0))
    relative = manager.relative_xy([1.0, 1.0], math.pi / 2.0, "odom")
    np.testing.assert_allclose(relative, [0.0, -1.0], atol=1e-6)

    manager.update(Goal2D("base_link", 0.5, -0.25, 11.0))
    np.testing.assert_allclose(manager.relative_xy([9.0, 9.0], 1.0, "odom"), [0.5, -0.25])


def test_lidar_preprocessing_is_41_rays_and_clipped():
    adapter = LidarRayAdapter("cpu")
    points = torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.0, 0.0], [9.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]])
    rays = adapter.project(points)
    assert tuple(rays.shape) == (1, 41)
    assert float(rays.min()) >= 0.1
    assert float(rays.max()) <= 5.0
    assert float(rays[0, 20]) == pytest.approx(0.2)


def test_sea_nav_observation_shape_order_and_history():
    obs = NavigationObservation(torch.device("cpu"))
    frame = obs.build(
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.tensor([[0.5, 0.0, 0.5]]),
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        torch.full((1, 41), 1.0),
        torch.tensor([[2.0, -1.0]]),
    )
    assert tuple(frame.shape) == (1, 550)
    # 55-D frame: gravity, scaled command, linear/angular velocity, log2 rays, goal.
    torch.testing.assert_close(frame[0, :12], torch.tensor([0.0, 0.0, -1.0, 1.0, 0.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(frame[0, -2:], torch.tensor([2.0, -1.0]))
    assert torch.count_nonzero(frame[0, 55:]).item() > 0


def test_himloco_observation_history_and_previous_action():
    obs = HIMLocoObservation(torch.device("cpu"))
    first = obs.build(torch.tensor([[0.5, 0.0, 0.5]]), torch.zeros(1, 3),
                      torch.tensor([[0.0, 0.0, -1.0]]), torch.zeros(1, 12), torch.zeros(1, 12))
    assert tuple(first.shape) == (1, 270)
    assert torch.count_nonzero(first[:, 45:]).item() == 0
    obs.record_action(torch.full((1, 12), 150.0))
    second = obs.build(torch.zeros(1, 3), torch.zeros(1, 3),
                       torch.tensor([[0.0, 0.0, -1.0]]), torch.zeros(1, 12), torch.zeros(1, 12))
    assert tuple(second.shape) == (1, 270)
    torch.testing.assert_close(second[0, 33:45], torch.full((12,), 100.0))


def test_joint_mapping_is_a_round_trip_permutation():
    forward = make_policy_to_motor()
    inverse = make_motor_to_policy()
    assert sorted(forward) == list(range(12))
    assert sorted(inverse) == list(range(12))
    assert [inverse[i] for i in forward] == list(range(12))


def test_stale_and_nan_safety_states():
    safety = SafetySupervisor(max_sensor_age=0.25)
    stale = safety.evaluate({"lowstate": None}, values=[np.zeros(3)])
    assert stale.state == RuntimeState.STALE_SENSOR
    invalid = safety.evaluate({"lowstate": 0.01}, values=[np.asarray([np.nan])])
    assert invalid.state == RuntimeState.INVALID_DATA
    emergency = safety.evaluate({"lowstate": 0.01}, values=[np.zeros(3)], wireless_emergency=True)
    assert emergency.state == RuntimeState.EMERGENCY_STOP


def test_nested_finite_values_and_sensor_log_are_bounded():
    assert _is_finite([np.zeros(3), {"q": np.ones(2)}])
    assert not _is_finite([np.zeros(3), {"q": np.asarray([np.nan])}])
    assert _health_for_log({"streams": {}, "finite_values": [np.zeros(1000)]}) == {"streams": {}}


def test_shadow_command_bridge_has_no_write_path():
    bridge = ReadOnlyCommandBridge([-1.0, -1.0, -2.0], [1.0, 1.0, 2.0])
    np.testing.assert_allclose(bridge.validate([2.0, 0.0, -3.0]), [1.0, 0.0, -2.0])
    bridge.assert_no_writes()
    with pytest.raises(RuntimeError, match="disabled"):
        bridge.send_low_level([0.0] * 12)
    assert bridge.write_count == 1


def test_command_filter_is_bounded_and_explicit():
    bridge = ReadOnlyCommandBridge([-1.0, -1.0, -2.0], [1.0, 1.0, 2.0], filter_alpha=0.5)
    np.testing.assert_allclose(bridge.filter([1.0, 0.0, 1.0]), [1.0, 0.0, 1.0])
    np.testing.assert_allclose(bridge.filter([-1.0, 0.0, -1.0]), [0.0, 0.0, 0.0])


def test_sensor_stream_summary_tracks_type_and_interarrival_fields():
    slot = Latest(message_type="unitree_go.msg.WirelessController")
    summary = slot.summary()
    assert summary["message_type"] == "unitree_go.msg.WirelessController"
    assert summary["gap_count"] == 0
    assert summary["stale_count"] == 0


def test_wireless_diagnostics_reads_nested_fields_without_ros_writes():
    class FakeWireless:
        __slots__ = ("keys", "lx", "nested")

        def __init__(self):
            self.keys = 3
            self.lx = 0.25
            self.nested = {"ignored": True}

    value = _ros_value(FakeWireless())
    assert value["keys"] == 3
    assert value["lx"] == 0.25


def test_diagnostics_source_has_no_ros_write_calls():
    source = Path("deploy/go2_onboard/diagnostics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "create_publisher" not in called_attributes
    assert "publish" not in called_attributes
    assert "send_low_level" not in called_attributes


def test_shadow_runtime_has_no_ros_write_calls():
    source = Path("deploy/go2_onboard/runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "LowCmd" not in imported_names | imported_from_names
    assert "create_publisher" not in called_attributes
    assert "publish" not in called_attributes


@pytest.mark.parametrize(
    "path,input_dim,output_dim",
    [
        ("artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt", 550, 3),
        ("models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt", 270, 12),
    ],
)
def test_deployed_model_contract(path, input_dim, output_dim):
    path = Path(path)
    if not path.exists():
        pytest.skip(path)
    if input_dim == 550:
        loaded = load_navigation_policy(str(path), "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json", torch.device("cpu"))
    else:
        loaded = load_himloco_policy(str(path), torch.device("cpu"))
    assert loaded.path == str(path.resolve())
    assert len(loaded.sha256) == 64
    with torch.inference_mode():
        output = loaded.policy(torch.zeros(1, input_dim))
    assert tuple(output.shape) == (1, output_dim)
    assert torch.isfinite(output).all()

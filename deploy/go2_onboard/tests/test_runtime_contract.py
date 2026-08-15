import math
import ast
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest
import torch

from deploy.go2_onboard.command_bridge import ReadOnlyCommandBridge
from deploy.go2_onboard.goal import Goal2D, GoalManager
from deploy.go2_onboard.himloco_observation import HIMLocoObservation
from deploy.go2_onboard.ipc_schema import decode_packet, encode_packet, make_packet
from deploy.go2_onboard.joint_mapping import make_motor_to_policy, make_policy_to_motor
from deploy.go2_onboard.lidar_ray_adapter import LidarRayAdapter
from deploy.go2_onboard.model_loader import load_himloco_policy, load_navigation_policy
from deploy.go2_onboard.navigation_observation import NavigationObservation
from deploy.go2_onboard.diagnostics import _ros_value
from deploy.go2_onboard.ros_state_reader import Latest, is_fresh
from deploy.go2_onboard.runtime import _health_for_log
from deploy.go2_onboard.safety_supervisor import RuntimeState, SafetySupervisor, _is_finite
from deploy.go2_onboard.sensor_bridge import duration_expired


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


def test_per_sensor_freshness_allows_nominal_lidar_period_but_not_timeout():
    assert is_fresh(0.002, 0.10)  # high-rate LowState
    assert is_fresh(0.072, 0.20)  # approximately one 13.8 Hz LiDAR period
    assert not is_fresh(0.101, 0.10)
    assert not is_fresh(0.201, 0.20)


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


def test_sensor_bridge_uses_continuous_executor_not_one_callback_per_packet():
    source = Path("deploy/go2_onboard/sensor_bridge.py").read_text(encoding="utf-8")
    assert "start_background_spin" in source
    assert "reader.spin_once" not in source
    reader_source = Path("deploy/go2_onboard/ros_state_reader.py").read_text(encoding="utf-8")
    assert "SingleThreadedExecutor" in reader_source
    assert "threading.RLock()" in reader_source


def test_sensor_bridge_duration_starts_after_worker_connection():
    assert not duration_expired(None, 15.0, 100.0)
    assert not duration_expired(100.0, 15.0, 114.999)
    assert duration_expired(100.0, 15.0, 115.0)
    assert not duration_expired(100.0, 0.0, 100000.0)

    source = Path("deploy/go2_onboard/sensor_bridge.py").read_text(encoding="utf-8")
    assert "started = None" in source
    assert "started = time.monotonic()" in source
    assert "connection, _ = server.accept()" in source


def test_split_shadow_ipc_packet_is_one_way_and_shape_checked():
    packet = make_packet(
        sequence=7,
        timestamp_monotonic=1.0,
        timestamp_wall=2.0,
        joint_pos=np.zeros(12),
        joint_vel=np.zeros(12),
        imu_ang_vel=np.zeros(3),
        projected_gravity=[0.0, 0.0, -1.0],
        base_linear_velocity_body=np.zeros(3),
        base_angular_velocity_body=np.zeros(3),
        lidar_rays=np.full(41, 5.0),
        goal_body=[2.0, 0.0],
        sensor_age={"lowstate": 0.01, "lidar": 0.02, "odom": 0.01, "goal": 0.01, "wireless": None},
        validity={"lowstate": True, "lidar": True, "odom": True, "goal": True},
    )
    decoded = decode_packet(encode_packet(packet))
    assert decoded["sequence"] == 7
    assert decoded["lidar_rays"] == [5.0] * 41
    assert "action" not in decoded
    assert "command" not in decoded
    invalid = dict(packet)
    invalid["goal_body"] = [float("nan"), 0.0]
    with pytest.raises(ValueError, match="goal_body"):
        encode_packet(invalid)


def test_split_shadow_modules_have_no_robot_write_symbols():
    for filename in ("sensor_bridge.py", "shadow_worker.py", "shadow_fixture.py"):
        source = Path("deploy/go2_onboard", filename).read_text(encoding="utf-8")
        assert "create_publisher" not in source
        assert "publish(" not in source
    worker_source = Path("deploy/go2_onboard/shadow_worker.py").read_text(encoding="utf-8")
    assert "from unitree_go.msg import LowCmd" not in worker_source
    assert "from unitree_go.msg import SportClient" not in worker_source
    assert "send_low_level" not in worker_source
    assert "sock.send" not in worker_source


def test_offline_split_shadow_ipc_smoke():
    repo_root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix="sea_nav_shadow_test_") as temp_dir:
        temp_dir = Path(temp_dir)
        socket_path = temp_dir / "shadow.sock"
        worker_log = temp_dir / "worker.jsonl"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(socket_path))
        server.listen(1)
        server.settimeout(10.0)
        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "deploy.go2_onboard.shadow_worker",
                "--socket",
                str(socket_path),
                "--log",
                str(worker_log),
                "--connect-timeout",
                "5",
            ],
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": str(repo_root)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            connection, _ = server.accept()
            packet_kwargs = dict(
                timestamp_monotonic=time.monotonic(),
                timestamp_wall=time.time(),
                joint_pos=[0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
                joint_vel=np.zeros(12),
                imu_ang_vel=np.zeros(3),
                projected_gravity=[0.0, 0.0, -1.0],
                base_linear_velocity_body=np.zeros(3),
                base_angular_velocity_body=np.zeros(3),
                lidar_rays=np.full(41, 5.0),
                goal_body=[2.0, 0.0],
                sensor_age={"lowstate": 0.01, "lidar": 0.01, "odom": 0.01, "goal": 0.01, "wireless": None},
                validity={"lowstate": True, "lidar": True, "odom": True, "goal": True},
            )
            for sequence in range(3):
                packet = make_packet(sequence=sequence, **packet_kwargs)
                connection.sendall(encode_packet(packet))
                time.sleep(0.02)
            connection.close()
            server.close()
            stdout, stderr = worker.communicate(timeout=15)
            assert worker.returncode == 0, stderr
            records = [json.loads(line) for line in worker_log.read_text(encoding="utf-8").splitlines()]
            assert len(records) == 3
            assert all(record["mode"] == "shadow_worker" for record in records)
            assert all(record["lowcmd_sent"] is False for record in records)
            assert records[-1]["sea_observation_shape"] == [1, 550]
            assert records[-1]["him_observation_shape"] == [1, 270]
        finally:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=5)
            try:
                server.close()
            except OSError:
                pass


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

import math
import ast
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pytest
import torch

from deploy.go2_onboard.command_bridge import ReadOnlyCommandBridge
from deploy.go2_onboard.goal import Goal2D, GoalManager
from deploy.go2_onboard.himloco_observation import HIMLocoObservation
from deploy.go2_onboard.ipc_schema import decode_packet, encode_packet, make_packet
from deploy.go2_onboard.joint_mapping import make_motor_to_policy, make_policy_to_motor
from deploy.go2_onboard.lidar_ray_adapter import LidarRayAdapter, LidarRayCache, NumpyLidarRayAdapter
from deploy.go2_onboard.model_loader import load_himloco_policy, load_navigation_policy
from deploy.go2_onboard.navigation_observation import NavigationObservation, clear_lidar_observation
from deploy.go2_onboard.diagnostics import _ros_value
from deploy.go2_onboard.ros_state_reader import Latest, is_fresh
from deploy.go2_onboard.runtime import _health_for_log
from deploy.go2_onboard.safety_supervisor import RuntimeState, SafetySupervisor, _is_finite
from deploy.go2_onboard.sensor_bridge import duration_expired
from deploy.go2_onboard.shadow_worker import dump_sea_observation
from deploy.go2_onboard.timing import FixedRate, RateStats


def test_go2_point_lio_uses_complete_transformed_raw_imu():
    config = Path("lio/sea_nav_lio_ws/src/point_lio_unilidar/config/sea_nav_go2.yaml")
    source = config.read_text(encoding="utf-8")
    assert 'imu_topic: "/sea_nav/lio/transformed_raw_imu"' in source
    assert 'imu_topic: "/sea_nav/lio/transformed_imu"' not in source

    transform = Path(
        "lio/sea_nav_lio_ws/src/transform_sensors/transform_sensors/transform_everything.py"
    ).read_text(encoding="utf-8")
    assert "self.imu_raw_pub.publish(transformed_imu)" in transform
    assert "transformed_imu.linear_acceleration.x = 0.0" in transform
    assert "transformed_imu.angular_velocity = transformed_angular_velocity" in transform


def test_lio_selfcheck_has_fail_closed_transformed_raw_imu_gate():
    source = Path("tools/go2_lio_selfcheck.sh").read_text(encoding="utf-8")
    assert 'check_imu_stream()' in source
    assert 'local topic="$1" label="$2" expected_frame="$3"' in source
    assert "check_imu_stream /utlidar/imu RAW_IMU" in source
    assert "check_transformed_raw_imu" in source
    assert "wait_for_transformed_raw_imu" in source
    assert "WAITING_TRANSFORMED_RAW_IMU" in source
    assert "TRANSFORMED_RAW_IMU_PUBLISHER=PASS" in source
    assert "RAW_IMU" in source
    assert "TRANSFORMED_RAW_IMU=PASS" in source
    assert "POINT_LIO_START_BLOCKED" in source
    assert "IMU_TIMESTAMP_MONOTONIC=PASS" in source
    assert source.index("check_imu_stream /utlidar/imu RAW_IMU") < source.index('start_or_reuse_point')
    point_start = source.index('start_terminal "Go2 Point-LIO')
    assert point_start < source.index("wait_for_transformed_raw_imu", point_start)


def test_lio_launcher_has_stale_supervisor_fail_closed_check():
    source = Path("tools/go2_nav_start.sh").read_text(encoding="utf-8")
    assert "STALE_SUPERVISOR" in source
    assert "existing_selfcheck_supervisors" in source


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


def test_lidar_self_filter_removes_measured_front_self_cluster_only():
    adapter = NumpyLidarRayAdapter()
    points = np.asarray([
        [0.63, 0.04, -0.24],  # measured center self-return region
        [0.42, 0.20, -0.24],  # measured side self-return region
        [0.35, 0.00, -0.24],  # real front obstacle, must remain
        [0.50, 0.00, -0.24],  # real front obstacle, must remain
        [4.00, 0.00, 0.00],   # distant point
    ], dtype=np.float32)
    rays = adapter.project(points)[0]
    assert rays[20] == pytest.approx(0.35)
    assert rays.shape == (41,)
    assert rays[20] < 1.0
    assert rays[20] == pytest.approx(0.35)


def test_lidar_self_filter_preserves_external_same_distance_point():
    adapter = NumpyLidarRayAdapter()
    points = np.asarray([[0.42, 0.30, -0.21]], dtype=np.float32)
    rays = adapter.project(points)[0]
    assert rays[int(round((math.radians(35.0) - (-2.0 * math.pi / 3.0)) / ((4.0 * math.pi / 3.0) / 40.0)))] == pytest.approx(0.516, abs=0.01)


def test_assume_clear_lidar_preserves_history_and_replaces_ray_encoding():
    observation = torch.arange(550, dtype=torch.float32).reshape(1, 550)
    cleared = clear_lidar_observation(observation)
    assert tuple(cleared.shape) == (1, 550)
    clear_value = torch.log2(torch.tensor(5.0))
    for frame in range(10):
        start = frame * 55
        torch.testing.assert_close(cleared[0, start + 12:start + 53], torch.full((41,), clear_value))
        torch.testing.assert_close(cleared[0, start:start + 12], observation[0, start:start + 12])
        torch.testing.assert_close(cleared[0, start + 53:start + 55], observation[0, start + 53:start + 55])


def test_lidar_cache_processes_each_message_once_and_reuses_snapshot():
    class CountingAdapter(NumpyLidarRayAdapter):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def project(self, points):
            self.calls += 1
            return super().project(points)

    adapter = CountingAdapter()
    cache = LidarRayCache(adapter)
    points = np.asarray([[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    expected = adapter.project(points)[0]
    adapter.calls = 0
    cache.update(points, received_at=12.5, source_timestamp=99.0)
    first = cache.snapshot()
    second = cache.snapshot()
    assert adapter.calls == 1
    np.testing.assert_allclose(first["rays"], expected)
    np.testing.assert_allclose(second["rays"], expected)
    assert first["processed_count"] == 1
    assert second["processed_count"] == 1
    assert first["received_at"] == pytest.approx(12.5)
    assert second["source_timestamp"] == pytest.approx(99.0)


def test_lidar_cache_new_message_updates_timestamp_and_projection():
    adapter = NumpyLidarRayAdapter()
    cache = LidarRayCache(adapter)
    cache.update(np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32), received_at=1.0, source_timestamp=10.0)
    before = cache.snapshot()
    cache.update(np.asarray([[1.5, 0.0, 0.0]], dtype=np.float32), received_at=2.0, source_timestamp=11.0)
    after = cache.snapshot()
    assert after["processed_count"] == before["processed_count"] + 1
    assert after["received_at"] == pytest.approx(2.0)
    assert after["source_timestamp"] == pytest.approx(11.0)
    assert after["rays"][20] == pytest.approx(1.5)


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


def test_sea_observation_dump_is_an_immutable_exact_policy_input(tmp_path):
    observation = torch.arange(550, dtype=torch.float32).reshape(1, 550)
    before = observation.clone()
    packet = {"sequence": 123, "goal_body": [2.0, 1.0]}
    output = tmp_path / "sea_observation.json"

    payload = dump_sea_observation(output, observation, packet)

    torch.testing.assert_close(observation, before, rtol=0.0, atol=0.0)
    assert payload["shape"] == [1, 550]
    assert payload["sea_observation"] == before.tolist()
    assert payload["sea_observation_finite"] is True
    assert payload["history_order"] == "oldest_to_newest"
    assert payload["frame_dim"] == 55
    assert payload["history_len"] == 10
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["sea_observation"] == before.tolist()


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
    assert "lidar_cache.snapshot()" in source
    assert ".project(" not in source
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


def test_forward_goal_status_contract_fixture():
    x, y, yaw, forward = -1.526054459, 0.343810399, 0.0, 0.50
    goal_x = x + forward * math.cos(math.radians(yaw))
    goal_y = y + forward * math.sin(math.radians(yaw))
    assert goal_x == pytest.approx(-1.026054459, abs=1e-9)
    assert goal_y == pytest.approx(0.343810399, abs=1e-9)

    yaw = math.radians(90.0)
    assert x + forward * math.cos(yaw) == pytest.approx(x, abs=1e-9)
    assert y + forward * math.sin(yaw) == pytest.approx(y + 0.50, abs=1e-9)

    selfcheck = Path("tools/go2_lio_selfcheck.sh").read_text(encoding="utf-8")
    launcher = Path("tools/go2_nav_start.sh").read_text(encoding="utf-8")
    assert "GOAL_MODE=FORWARD" in selfcheck
    assert "CURRENT_YAW_DEG=%s" in selfcheck
    assert "rclpy.spin_once" in selfcheck
    assert "FORWARD_ODOM_READ_ATTEMPTS" in selfcheck
    assert "contextlib.redirect_stdout" in selfcheck
    assert "FORWARD_ODOM_PARSE_FAILED" in selfcheck
    assert "LAUNCH_PGID" in selfcheck
    assert "setsid ros2 launch" in selfcheck
    assert "point_lio_process_tree" in selfcheck


def test_lio_cleanup_distinguishes_live_processes_and_zombies():
    selfcheck = Path("tools/go2_lio_selfcheck.sh").read_text(encoding="utf-8")
    launcher = Path("tools/go2_nav_start.sh").read_text(encoding="utf-8")
    assert "POINT_LIO_GROUP_AFTER_SIGINT" in selfcheck
    assert "POINT_LIO_GROUP_AFTER_SIGTERM" in selfcheck
    assert "POINT_LIO_GROUP_AFTER_SIGKILL" in selfcheck
    assert "ZOMBIE_WAITING_REAP" in selfcheck
    assert "STAT=$stat" in selfcheck
    assert "SELF_CHECK_CLEANUP=FAIL" in launcher
    assert "CLEAN_SHUTDOWN=FAIL" in launcher
    assert "GOAL_STATUS_FILE=" in launcher
    assert "GOAL_STATUS_CONTENT_BEGIN" in launcher


def test_sensor_bridge_handles_signal_shutdown_without_executor_traceback():
    bridge_source = Path("deploy/go2_onboard/sensor_bridge.py").read_text(encoding="utf-8")
    reader_source = Path("deploy/go2_onboard/ros_state_reader.py").read_text(encoding="utf-8")
    assert "signal.SIGTERM" in bridge_source
    assert "stop_requested" in bridge_source
    assert "bridge.close()" in bridge_source
    assert "ExternalShutdownException" in reader_source
    assert "def _spin_loop" in reader_source
    assert "spin_thread.join" in reader_source


def test_ros_reader_shutdown_smoke_is_ordered_and_idempotent():
    from deploy.go2_onboard.ros_state_reader import RosStateReader

    class FakeExternalShutdown(Exception):
        pass

    class FakeExecutor:
        def __init__(self):
            self.stop = threading.Event()

        def add_node(self, _node):
            pass

        def spin(self):
            self.stop.wait(2.0)

        def shutdown(self, timeout_sec=1.0):
            assert timeout_sec == 1.0
            self.stop.set()

    class FakeNode:
        def destroy_node(self):
            pass

    reader = RosStateReader.__new__(RosStateReader)
    reader._closed = False
    reader._close_requested = False
    reader._spin_error = None
    reader._executor = None
    reader._spin_thread = None
    reader._executor_type = FakeExecutor
    reader._external_shutdown_exception = FakeExternalShutdown
    reader.node = FakeNode()
    reader._owns_rclpy = False
    reader._rclpy = None

    reader.start_background_spin()
    assert reader._spin_thread.is_alive()
    reader.close()
    reader.close()
    assert reader._spin_thread is None
    assert reader._executor is None


def test_timing_stats_report_target_rate_and_percentiles():
    stats = RateStats()
    stats.observe(0.00)
    stats.observe(0.02)
    stats.observe(0.04)
    summary = stats.summary()
    assert summary["mean_rate_hz"] == pytest.approx(50.0)
    assert summary["p50_period_ms"] == pytest.approx(20.0)
    assert summary["p95_period_ms"] == pytest.approx(20.0)


def test_fixed_rate_uses_absolute_deadlines_not_period_plus_work():
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = FakeClock()
    rate = FixedRate(50.0, clock=clock.monotonic, sleeper=clock.sleep)
    starts = [clock.now]
    results = []
    for _ in range(5):
        clock.now += 0.002  # simulated 2 ms of work
        results.append(rate.sleep())
        starts.append(clock.now)
    periods = np.diff(starts)
    np.testing.assert_allclose(periods, np.full(5, 0.020), atol=1e-12)
    assert all(not result["deadline_miss"] for result in results)


def test_fixed_rate_reports_overrun_and_resynchronizes_without_drift():
    class FakeClock:
        def __init__(self):
            self.now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = FakeClock()
    rate = FixedRate(50.0, clock=clock.monotonic, sleeper=clock.sleep)
    clock.now += 0.025
    overrun = rate.sleep()
    assert overrun["deadline_miss"] is True
    assert overrun["late_s"] == pytest.approx(0.005)
    clock.now += 0.002
    next_tick = rate.sleep()
    assert next_tick["deadline_miss"] is False
    assert clock.now == pytest.approx(0.045)


def test_shadow_logs_timing_breakdown_without_per_packet_stdout_or_flush():
    for filename in ("sensor_bridge.py", "shadow_worker.py"):
        source = Path("deploy/go2_onboard", filename).read_text(encoding="utf-8")
        assert ".flush()" not in source
        assert "print(json.dumps(record" not in source


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
        sea_snapshot = temp_dir / "sea_observation.json"
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
                "--dump-sea-observation",
                str(sea_snapshot),
                "--dump-sea-observation-after-samples",
                "2",
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
            snapshot = json.loads(sea_snapshot.read_text(encoding="utf-8"))
            assert snapshot["sequence"] == 1
            assert snapshot["shape"] == [1, 550]
            assert len(snapshot["sea_observation"]) == 1
            assert len(snapshot["sea_observation"][0]) == 550
            assert snapshot["sea_observation_finite"] is True
            for field in (
                "ipc_receive_wait_ms", "decode_ms", "sea_obs_build_ms", "sea_inference_latency_ms",
                "him_obs_build_ms", "him_inference_latency_ms", "loop_total_ms", "log_write_ms",
            ):
                assert field in records[-1]
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

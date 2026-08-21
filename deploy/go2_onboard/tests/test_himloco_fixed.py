import ast
from collections import deque
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

from deploy.go2_onboard.himloco_fixed_control import (
    ACTION_SCALE,
    APPROVED_POLICIES,
    approved_policy_profile_for_path,
    COMMAND_SCALE,
    DEFAULT_ANGLES,
    EXPECTED_INPUT_DIM,
    EXPECTED_OUTPUT_DIM,
    EXPECTED_SHA256,
    FixedHIMLocoController,
    ARM_WAIT_TIMEOUT,
    POLICY_WARMUP_STEPS,
    KD,
    LEGACY_POLICY_1_SHA256,
    KP,
    POLICY_TO_MOTOR,
    LowStateWatchdog,
    LowStateSnapshot,
    RuntimeState,
    SafetyError,
    build_observation,
    build_target_q,
    configure_torch_runtime,
    require_performance_governor,
    history_repeat_on_first_for_profile,
    advance_active_deadline,
    pose_transition_target,
    projected_gravity_from_wxyz,
    sport_mode_allows_low_level,
    low_level_gate_allows,
    lowstate_snapshot_is_newer,
    sha256_file,
    validate_fixed_command,
    advance_deadline,
)


def test_sport_mode_gate_uses_read_only_service_status():
    assert sport_mode_allows_low_level(1)
    assert not sport_mode_allows_low_level(0)
    assert not low_level_gate_allows(1, "mcf")
    assert low_level_gate_allows(1, "")
    assert not low_level_gate_allows(0, "")
from deploy.go2_onboard.himloco_observation import HIMLocoObservation


def test_current_1460_hash_and_shape():
    path = Path("models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt")
    assert sha256_file(str(path)) == EXPECTED_SHA256
    policy = torch.jit.load(str(path), map_location="cpu").eval()
    with torch.inference_mode():
        output = policy(torch.zeros((1, EXPECTED_INPUT_DIM)))
    assert tuple(output.shape) == (1, EXPECTED_OUTPUT_DIM)
    assert POLICY_WARMUP_STEPS == 10


def test_approved_policy_allowlist_contains_only_two_named_hashes():
    assert APPROVED_POLICIES == {
        "himloco_1460": EXPECTED_SHA256,
        "legacy_policy_1": LEGACY_POLICY_1_SHA256,
    }


@pytest.mark.parametrize(
    ("profile", "path", "expected_sha"),
    [
        (
            "himloco_1460",
            "models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt",
            EXPECTED_SHA256,
        ),
        ("legacy_policy_1", "models/locomotion/himloco/policy_1.pt", LEGACY_POLICY_1_SHA256),
    ],
)
def test_approved_policy_sha_and_shape_pass(profile, path, expected_sha, capsys):
    controller = object.__new__(FixedHIMLocoController)
    controller.args = SimpleNamespace(policy=path)
    controller.validate_model()
    assert controller.policy_profile == profile
    assert sha256_file(path) == expected_sha
    assert "input=270 output=12" in capsys.readouterr().out


def test_unknown_sha_fails_before_lowcmd_transport(monkeypatch):
    controller = object.__new__(FixedHIMLocoController)
    controller.args = SimpleNamespace(policy="models/locomotion/himloco/policy_1.pt")
    monkeypatch.setattr(
        "deploy.go2_onboard.himloco_fixed_control.sha256_file",
        lambda _: "0" * 64,
    )
    with pytest.raises(SafetyError, match="not approved"):
        controller.validate_model()
    assert not hasattr(controller, "publisher")
    assert not hasattr(controller, "low_cmd")


def test_approved_hash_with_wrong_shape_fails(monkeypatch, tmp_path):
    wrong_policy = tmp_path / "wrong_shape.pt"
    module = torch.nn.Linear(EXPECTED_INPUT_DIM, EXPECTED_OUTPUT_DIM - 1)
    traced = torch.jit.trace(module.eval(), torch.zeros((1, EXPECTED_INPUT_DIM)))
    traced.save(str(wrong_policy))
    controller = object.__new__(FixedHIMLocoController)
    controller.args = SimpleNamespace(policy=str(wrong_policy))
    monkeypatch.setattr(
        "deploy.go2_onboard.himloco_fixed_control.sha256_file",
        lambda _: EXPECTED_SHA256,
    )
    with pytest.raises(SafetyError, match="270->12"):
        controller.validate_model()


def test_policy_override_does_not_bypass_shape_gate(tmp_path):
    wrong_policy = tmp_path / "override_wrong_shape.pt"
    module = torch.nn.Linear(EXPECTED_INPUT_DIM, EXPECTED_OUTPUT_DIM - 1)
    traced = torch.jit.trace(module.eval(), torch.zeros((1, EXPECTED_INPUT_DIM)))
    traced.save(str(wrong_policy))
    controller = object.__new__(FixedHIMLocoController)
    controller.args = SimpleNamespace(
        policy=str(wrong_policy),
        allow_policy_override=True,
    )
    with pytest.raises(SafetyError, match="270->12"):
        controller.validate_model()


def test_fixed_observation_is_270_and_newest_first():
    him = HIMLocoObservation(torch.device("cpu"))
    observation = build_observation(
        him, [0.1, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0],
        DEFAULT_ANGLES, [0.0] * 12,
    )
    assert tuple(observation.shape) == (1, 270)
    torch.testing.assert_close(observation[0, :3], torch.tensor([0.2, 0.0, 0.025]))
    assert torch.count_nonzero(observation[0, 45:]) == 0


def test_policy_history_can_be_primed_from_latest_pose():
    him = HIMLocoObservation(torch.device("cpu"))
    observation = build_observation(
        him, [0.1, 0.0, 0.1], [0.01, 0.02, 0.03], [0.0, 0.0, -1.0],
        DEFAULT_ANGLES + 0.01, [0.1] * 12, repeat_history=True,
    )
    frames = observation.reshape(1, 6, 45)
    for index in range(1, 6):
        torch.testing.assert_close(frames[:, 0], frames[:, index])


def test_legacy_history_starts_current_frame_then_five_zero_frames():
    him = HIMLocoObservation(torch.device("cpu"))
    command = [0.4, 0.0, 0.0]
    gyro = [0.01, -0.02, 0.03]
    gravity = [0.0, 0.0, -1.0]
    q = DEFAULT_ANGLES + 0.01
    dq = [0.1] * 12
    assert history_repeat_on_first_for_profile("legacy_policy_1") is False
    observation = build_observation(
        him, command, gyro, gravity, q, dq,
        repeat_history=history_repeat_on_first_for_profile("legacy_policy_1"),
    )
    frames = observation.reshape(1, 6, 45)
    assert torch.count_nonzero(frames[:, 1:]) == 0

    him.record_action(torch.ones((1, 12)))
    second = build_observation(him, command, gyro, gravity, q, dq, repeat_history=False)
    second_frames = second.reshape(1, 6, 45)
    torch.testing.assert_close(second_frames[:, 1], frames[:, 0])
    assert torch.count_nonzero(second_frames[:, 2:]) == 0


def test_1460_history_initialization_remains_repeat_current():
    assert history_repeat_on_first_for_profile("himloco_1460") is True


def test_legacy_profile_matches_old_deploy_observation_and_action_fixture():
    command = np.asarray([0.4, 0.0, 0.0], dtype=np.float32)
    gyro = np.asarray([0.12, -0.07, 0.20], dtype=np.float32)
    gravity = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    q = DEFAULT_ANGLES + np.linspace(-0.03, 0.03, 12, dtype=np.float32)
    dq = np.linspace(-0.1, 0.1, 12, dtype=np.float32)
    previous_action = np.zeros(12, dtype=np.float32)

    old_frame = np.empty(45, dtype=np.float32)
    old_frame[0:3] = command * np.asarray([2.0, 2.0, 0.25], dtype=np.float32)
    old_frame[3:6] = gyro * 0.25
    old_frame[6:9] = gravity
    old_frame[9:21] = q - DEFAULT_ANGLES
    old_frame[21:33] = dq * 0.05
    old_frame[33:45] = previous_action
    old_observation = np.concatenate(
        [old_frame, np.zeros(5 * 45, dtype=np.float32)]
    ).reshape(1, 270)

    him = HIMLocoObservation(torch.device("cpu"))
    current_observation = build_observation(
        him, command, gyro, gravity, q, dq,
        repeat_history=history_repeat_on_first_for_profile("legacy_policy_1"),
    )
    torch.testing.assert_close(current_observation, torch.from_numpy(old_observation))

    policy = torch.jit.load("models/locomotion/himloco/policy_1.pt", map_location="cpu").eval()
    with torch.inference_mode():
        old_action = policy(torch.from_numpy(old_observation))
        current_action = policy(current_observation)
    torch.testing.assert_close(current_action, old_action)


def _handoff_controller(profile="legacy_policy_1", action=None):
    controller = object.__new__(FixedHIMLocoController)
    controller.policy_profile = profile
    controller.him_obs = HIMLocoObservation(torch.device("cpu"))
    controller.low_cmd = SimpleNamespace(
        motor_cmd=[SimpleNamespace(q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(12)]
    )
    controller.policy = action or (lambda observation: torch.zeros((1, 12)))
    controller.last_forward_ms = 0.0
    controller.last_action_ms = 0.0
    controller.last_prepare_ms = 0.0
    controller.last_observation_ms = 0.0
    controller.forward_durations = deque()
    controller.prepare_durations = deque()
    controller.observation_durations = deque()
    controller.action_durations = deque()
    controller.previous_action = np.zeros(12, dtype=np.float32)
    return controller


def _handoff_snapshot(received_at):
    message = SimpleNamespace(
        motor_state=[SimpleNamespace(q=float(q), dq=0.0) for q in DEFAULT_ANGLES],
        imu_state=SimpleNamespace(gyroscope=[0.0, 0.0, 0.0], quaternion=[1.0, 0.0, 0.0, 0.0]),
    )
    return LowStateSnapshot(message=message, received_at=received_at, remote_keys=0)


def test_first_active_observation_is_forwarded_without_second_history_push():
    seen = []

    def policy(observation):
        seen.append(observation.detach().clone())
        return torch.zeros((1, 12))

    controller = _handoff_controller(action=policy)
    snapshot = _handoff_snapshot(2.0)
    first = controller._initialize_policy_history(snapshot, SimpleNamespace(as_array=lambda: np.zeros(3, dtype=np.float32)))
    history_before_forward = controller.him_obs.history.buffer.clone()

    controller._run_policy_once(
        SimpleNamespace(as_array=lambda: np.zeros(3, dtype=np.float32)),
        snapshot,
        observation=first,
    )

    torch.testing.assert_close(seen[0], first)
    torch.testing.assert_close(controller.him_obs.history.buffer, history_before_forward)


def test_second_active_cycle_performs_one_history_shift():
    seen = []

    def policy(observation):
        seen.append(observation.detach().clone())
        return torch.ones((1, 12))

    controller = _handoff_controller(action=policy)
    command = SimpleNamespace(as_array=lambda: np.zeros(3, dtype=np.float32))
    snapshot = _handoff_snapshot(2.0)
    first = controller._initialize_policy_history(snapshot, command)
    controller._run_policy_once(command, snapshot, observation=first)
    first_frame = first.reshape(1, 6, 45)[:, 0].clone()

    controller._run_policy_once(command, _handoff_snapshot(3.0))
    frames = controller.him_obs.history.buffer
    torch.testing.assert_close(frames[:, 1], first_frame)
    torch.testing.assert_close(frames[:, 0, 33:45], torch.ones((1, 12)))
    assert len(seen) == 2


def test_a_snapshot_requires_newer_lowstate_timestamp():
    baseline = _handoff_snapshot(10.0)
    same = _handoff_snapshot(10.0)
    newer = _handoff_snapshot(10.001)
    assert not lowstate_snapshot_is_newer(same, baseline)
    assert lowstate_snapshot_is_newer(newer, baseline)

    controller = object.__new__(FixedHIMLocoController)
    snapshots = iter((same, newer))
    controller._snapshot = lambda: next(snapshots)
    controller._check_runtime_safety = lambda snapshot: None
    result = controller._wait_for_fresh_policy_snapshot(baseline)
    assert result.received_at > baseline.received_at


def test_joint_mapping_and_target_contract():
    assert POLICY_TO_MOTOR == (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
    action = np.arange(12, dtype=np.float32)
    np.testing.assert_allclose(build_target_q(action), DEFAULT_ANGLES + ACTION_SCALE * action)
    assert KP == 20.0
    assert KD == 0.5
    np.testing.assert_allclose(COMMAND_SCALE, [2.0, 2.0, 0.25])


def test_pose_transition_starts_at_current_and_ends_at_default():
    initial = DEFAULT_ANGLES + np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    first = pose_transition_target(initial, 0, 100)
    last = pose_transition_target(initial, 99, 100)
    np.testing.assert_allclose(first, initial)
    np.testing.assert_allclose(last, DEFAULT_ANGLES)
    previous = first
    for step in range(1, 100):
        current = pose_transition_target(initial, step, 100)
        assert np.max(np.abs(current - previous)) < 0.01
        previous = current


@pytest.mark.parametrize("command", [(0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.0, 0.0, 0.15), (0.0, 0.0, -0.15)])
def test_fixed_command_whitelist(command):
    assert validate_fixed_command(*command).as_array().shape == (3,)


def test_legacy_reproduction_requires_exact_approved_command():
    command = validate_fixed_command(
        0.4,
        0.0,
        0.0,
        policy_profile="legacy_policy_1",
        legacy_repro=True,
    )
    np.testing.assert_allclose(command.as_array(), [0.4, 0.0, 0.0])


@pytest.mark.parametrize("command", [(0.2, 0.0, 0.0), (0.3, 0.0, 0.0), (0.5, 0.0, 0.0)])
def test_legacy_reproduction_rejects_other_speeds(command):
    with pytest.raises(SafetyError, match="exact command"):
        validate_fixed_command(
            *command,
            policy_profile="legacy_policy_1",
            legacy_repro=True,
        )


def test_legacy_reproduction_rejects_1460_profile():
    with pytest.raises(SafetyError, match="requires the approved legacy_policy_1"):
        validate_fixed_command(
            0.4,
            0.0,
            0.0,
            policy_profile="himloco_1460",
            legacy_repro=True,
        )


def test_legacy_reproduction_requires_approved_file_content():
    assert approved_policy_profile_for_path("models/locomotion/himloco/policy_1.pt") == "legacy_policy_1"
    unknown = Path("/tmp/unknown_himloco_policy.pt")
    unknown.write_bytes(b"not-an-approved-policy")
    with pytest.raises(SafetyError, match="not approved"):
        approved_policy_profile_for_path(str(unknown))
    unknown.unlink()


def test_normal_mode_still_rejects_legacy_speed():
    with pytest.raises(SafetyError, match="conservative first-test limit"):
        validate_fixed_command(0.4, 0.0, 0.0)


@pytest.mark.parametrize("command", [(-0.01, 0.0, 0.0), (0.16, 0.0, 0.0), (0.0, 0.01, 0.0), (0.1, 0.0, 0.1)])
def test_fixed_command_whitelist_rejects_unsafe_modes(command):
    with pytest.raises(SafetyError):
        validate_fixed_command(*command)


def test_watchdog_and_imu_finite_contract():
    watchdog = LowStateWatchdog(0.10)
    assert watchdog.fresh(10.0, 10.09)
    assert not watchdog.fresh(10.0, 10.11)
    np.testing.assert_allclose(projected_gravity_from_wxyz([1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, -1.0])


def test_performance_governor_preflight_accepts_all_performance_policies(capsys):
    require_performance_governor({"policy0": "performance", "policy1": "performance"})
    assert "CPU governor preflight: performance" in capsys.readouterr().out


def test_performance_governor_preflight_rejects_powersave():
    with pytest.raises(SafetyError, match="must be 'performance'"):
        require_performance_governor({"policy0": "powersave"})


def test_performance_governor_preflight_rejects_missing_policies():
    with pytest.raises(SafetyError, match="preflight unavailable"):
        require_performance_governor({})


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
    assert "self.lowstate_subscriber.Init(self._low_state_callback, 0)" in source
    assert "self.lowstate_subscriber.Init(self._low_state_callback, 10)" not in source
    assert ARM_WAIT_TIMEOUT == 5.0
    assert "no fresh LowState received within" in source


def test_startup_sequence_has_pose_transition_and_policy_gate():
    source = Path("deploy/go2_onboard/himloco_fixed_control.py").read_text(encoding="utf-8")
    assert "POSE_TRANSITION_START" in source
    assert "POSE_TRANSITION_COMPLETE" in source
    assert "DEFAULT_POSE_HOLD" in source
    assert "HISTORY_INIT_SOURCE=latest_default_pose_lowstate" in source
    assert "history_repeat_on_first_for_profile" in source
    assert "self.lowstate_subscriber.Init(self._low_state_callback, 0)" in source
    assert "self.policy_durations" in source
    assert "self.control_periods" in source
    assert "deadline_miss_count" in source
    assert "advance_active_deadline" in source
    assert "period_p99_ms" in source
    assert "publish_p95_ms" in source
    assert "callback_p95_ms" in source
    assert "forward_p95_ms" in source
    assert "self._dump_event_trace" in source
    assert "self.lowstate_subscriber.Init(self._low_state_callback, 0)" in source


def test_comm_only_mode_skips_policy_load_and_execution():
    source = Path("deploy/go2_onboard/himloco_fixed_control.py").read_text(encoding="utf-8")
    assert "--comm-only" in source
    assert "if not args.comm_only:" in source
    assert "controller.run_comm_only()" in source


def test_absolute_deadline_does_not_add_work_time_and_resyncs_overrun():
    next_tick, sleep_time = advance_deadline(0.0, 0.002)
    assert next_tick == pytest.approx(0.02)
    assert sleep_time == pytest.approx(0.018)

    next_tick, sleep_time = advance_deadline(next_tick, 0.022)
    assert next_tick == pytest.approx(0.04)
    assert sleep_time == pytest.approx(0.018)

    next_tick, sleep_time = advance_deadline(next_tick, 0.105)
    assert next_tick == pytest.approx(0.105)
    assert sleep_time == 0.0


def test_active_scheduler_path_stays_at_50hz_for_1000_cycles():
    """Exercise the same post-control scheduling order used by ACTIVE.run."""
    now = 0.0
    next_tick = 0.0
    starts = []
    lateness = []
    for _ in range(1000):
        starts.append(now)
        now += 0.001  # representative control work; below the 20 ms budget
        next_tick, sleep_time, late = advance_active_deadline(next_tick, now)
        lateness.append(late)
        now += sleep_time

    periods_ms = np.diff(np.asarray(starts)) * 1000.0
    assert np.percentile(periods_ms, 50) == pytest.approx(20.0)
    assert np.percentile(periods_ms, 95) == pytest.approx(20.0)
    assert np.max(periods_ms) == pytest.approx(20.0)
    assert max(lateness) == pytest.approx(0.0)


def test_active_run_advances_deadline_once():
    source = Path("deploy/go2_onboard/himloco_fixed_control.py").read_text(encoding="utf-8")
    active = source.split("    def run(self):", 1)[1].split("    def run_comm_only", 1)[0]
    assert "next_tick += CONTROL_DT" not in active
    assert active.count("advance_active_deadline(") == 1


def test_realtime_torch_defaults_are_single_threaded():
    source = Path("deploy/go2_onboard/himloco_fixed_control.py").read_text(encoding="utf-8")
    assert "TORCH_THREADS = 1" in source
    assert "TORCH_INTEROP_THREADS = 1" in source
    assert callable(configure_torch_runtime)
    transition = source.split("    def _move_to_default_pos", 1)[1].split("    def _wait_for_policy_arm", 1)[0]
    assert "self.policy" not in transition


def test_lowstate_callback_updates_thread_safe_receive_metrics():
    class Message:
        wireless_remote = bytes(24)

    controller = object.__new__(FixedHIMLocoController)
    controller.snapshot_lock = __import__("threading").RLock()
    controller.snapshot = type("Snapshot", (), {})()
    controller.last_lowstate_rx = 0.0
    controller.lowstate_rx_count = 0
    controller.lowstate_intervals = []
    controller.lowstate_callback_durations = []
    controller.event_trace = []
    controller._low_state_callback(Message())
    assert controller.lowstate_rx_count == 1
    assert controller.snapshot.message.__class__ is Message
    assert controller.snapshot.received_at > 0.0


def test_wait_for_pose_arm_keeps_waiting_after_first_fresh_lowstate():
    source = Path("deploy/go2_onboard/himloco_fixed_control.py").read_text(encoding="utf-8")
    wait_block = source.split("    def _wait_for_pose_arm", 1)[1].split("    def _move_to_default_pos", 1)[0]
    assert "if not self.watchdog.fresh" in wait_block
    assert "continue" in wait_block
    assert "if key_pressed(snapshot.remote_keys, self.POSE_ARM_KEY)" in wait_block


def test_offline_benchmark_is_no_write_and_uses_current_policy_path():
    source = Path("deploy/go2_onboard/himloco_offline_benchmark.py").read_text(encoding="utf-8")
    assert "unitree_sdk2py" not in source
    assert "ChannelPublisher" not in source
    assert "LowCmd" not in source
    assert "--steps" in source

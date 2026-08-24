import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import deploy.go2_onboard.sea_nav_himloco_navigation as navigation_module

from deploy.go2_onboard.sea_nav_himloco_navigation import (
    HIM_1460_SHA256,
    NAV_LOWER,
    NAV_UPPER,
    NavigationLimiter,
    NavigationMailbox,
    NavigationProcess,
    SeaNavHimLocoController,
    _navigation_result,
    goal_is_reached,
    validate_navigation_himloco_policy,
    validate_navigation_model,
)
from deploy.go2_onboard.sea_nav_sport_navigation import official_mpc_contract_probe
from deploy.go2_onboard.himloco_fixed_control import (
    FixedCommand,
    FixedHIMLocoController,
    RuntimeState,
    SafetyError,
)


def test_navigation_model_contract_and_hash():
    loaded = validate_navigation_model(
        "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt",
        "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json",
    )
    assert loaded.sha256 == "d1242c74ff56189f20d7a12948d86078287651cd1f70215d9c308d04f4b561de"


def test_official_mpc_probe_uses_unit_projected_gravity_and_full_observation():
    loaded = validate_navigation_model(
        "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt",
        "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json",
    )
    probe = official_mpc_contract_probe(loaded)
    assert probe["observation_shape"] == [1, 550]
    assert probe["gravity_convention"] == "unit_projected_gravity_z=-1"
    assert len(probe["probe_action"]) == 3
    assert np.isfinite(probe["probe_action"]).all()


def test_navigation_limits_allow_forward_arc_and_force_no_lateral_motion():
    limiter = NavigationLimiter(filter_alpha=1.0)
    raw, limited, safe = limiter.apply([0.4, 0.8, -0.4])
    np.testing.assert_allclose(raw, [0.4, 0.8, -0.4])
    np.testing.assert_allclose(limited, [0.4, 0.0, -0.4])
    np.testing.assert_allclose(safe, [0.4, 0.0, -0.4])
    np.testing.assert_allclose(NAV_LOWER, [0.0, -np.inf, -np.inf])
    np.testing.assert_allclose(NAV_UPPER, [np.inf, 0.0, np.inf])


def test_navigation_vx_max_is_a_limiter_bound_not_a_fixed_command():
    limiter = NavigationLimiter(filter_alpha=1.0, vx_max=0.05)
    raw, limited, safe = limiter.apply([0.4, 0.0, 0.0])
    np.testing.assert_allclose(raw, [0.4, 0.0, 0.0])
    np.testing.assert_allclose(limited, [0.05, 0.0, 0.0])
    np.testing.assert_allclose(safe, [0.05, 0.0, 0.0])


def test_navigation_vy_can_be_explicitly_enabled_for_2d_experiment():
    limiter = NavigationLimiter(filter_alpha=1.0, vx_max=0.1, vy_max=0.1)
    raw, limited, safe = limiter.apply([0.1, 0.2, 0.0])
    np.testing.assert_allclose(raw, [0.1, 0.2, 0.0])
    np.testing.assert_allclose(limited, [0.1, 0.1, 0.0])
    np.testing.assert_allclose(safe, [0.1, 0.1, 0.0])


def test_navigation_vx_max_can_be_unlimited():
    limiter = NavigationLimiter(filter_alpha=1.0, vx_max=float("inf"))
    _, limited, safe = limiter.apply([2.0, 0.0, 0.0])
    np.testing.assert_allclose(limited, [2.0, 0.0, 0.0])
    np.testing.assert_allclose(safe, [2.0, 0.0, 0.0])


def test_navigation_worker_config_dict_reaches_freshness_gate():
    packet = {
        "sequence": 1,
        "timestamp_monotonic": time.monotonic(),
        "validity": {"lowstate": False, "lidar": False, "odom": False, "goal": False},
        "sensor_age": {"lowstate": 1.0, "lidar": 1.0, "odom": 1.0},
        "goal_body": [0.5, 0.0],
    }
    config = {
        "assume_clear_lidar": False,
        "lowstate_max_age": 0.10,
        "odom_max_age": 0.10,
        "lidar_max_age": 0.20,
        "goal_tolerance": 0.30,
    }
    resettable = SimpleNamespace(reset=lambda: None)
    config["goal_reached_confirmations"] = 5
    result, reached, streak = _navigation_result(
        packet, None, resettable, resettable, np.zeros(3), config, False
    )
    assert result["runtime_state"] == "STALE_NAVIGATION"
    assert reached is False
    assert streak == 0


def test_goal_reached_requires_consecutive_fresh_confirmations(monkeypatch):
    monkeypatch.setattr(
        navigation_module,
        "infer",
        lambda sea, observation, output_dim: torch.zeros((1, 3), dtype=torch.float32),
    )
    packet = {
        "sequence": 1,
        "timestamp_monotonic": time.monotonic(),
        "validity": {"lowstate": True, "lidar": True, "odom": True, "goal": True},
        "sensor_age": {"lowstate": 0.01, "lidar": 0.01, "odom": 0.01},
        "goal_body": [0.10, 0.0],
        "projected_gravity": [0.0, 0.0, -9.81],
        "base_linear_velocity_body": [0.0, 0.0, 0.0],
        "base_angular_velocity_body": [0.0, 0.0, 0.0],
        "lidar_rays": [5.0] * 5,
    }
    config = {
        "assume_clear_lidar": True,
        "lowstate_max_age": 0.10,
        "odom_max_age": 0.10,
        "lidar_max_age": 0.20,
        "goal_tolerance": 0.15,
        "goal_reached_confirmations": 3,
        "navigation_vx_max": 0.15,
        "navigation_vy_max": 0.0,
    }
    resettable = SimpleNamespace(
        reset=lambda: None,
        build=lambda *args: torch.zeros((1, 550), dtype=torch.float32),
    )
    streak = 0
    reached = False
    for index in range(2):
        packet["sequence"] = index
        result, reached, streak = _navigation_result(
            packet, None, resettable, resettable, np.zeros(3), config, reached, streak
        )
        assert result["runtime_state"] == "NAVIGATION"
        assert reached is False
    result, reached, streak = _navigation_result(
        packet, None, resettable, resettable, np.zeros(3), config, reached, streak
    )
    assert result["runtime_state"] == "GOAL_REACHED"
    assert reached is True
    assert streak == 3


def test_navigation_limiter_rejects_nonfinite_output():
    with pytest.raises(FloatingPointError, match="finite 3-vector"):
        NavigationLimiter().apply([0.0, float("nan"), 0.0])


def test_navigation_stale_mailbox_fails_closed():
    mailbox = NavigationMailbox(max_age=0.1)
    command, reason, age = mailbox.current()
    assert command.as_array().tolist() == [0.0, 0.0, 0.0]
    assert reason == "navigation_command_stale"
    assert age == float("inf")

    mailbox.update(
        {
            "command": [0.1, 0.0, -0.1],
            "runtime_state": "NAVIGATION",
            "sensor_sequence": 1,
            "sensor_timestamp_monotonic": time.monotonic(),
        }
    )
    command, reason, age = mailbox.current()
    np.testing.assert_allclose(command.as_array(), [0.1, 0.0, -0.1])
    assert reason == ""
    assert 0.0 <= age < 0.1


def test_stale_sensor_command_is_forced_to_zero():
    mailbox = NavigationMailbox(max_age=0.25)
    mailbox.update(
        {
            "command": [0.15, 0.0, 0.0],
            "runtime_state": "NAVIGATION",
            "sensor_sequence": 10,
            "sensor_timestamp_monotonic": time.monotonic() - 1.0,
        }
    )
    command, reason, _ = mailbox.current()
    np.testing.assert_allclose(command.as_array(), [0.0, 0.0, 0.0])
    assert reason == "navigation_command_stale"


def test_active_handoff_skips_prearm_backlog_until_fresh_packet():
    mailbox = NavigationMailbox(max_age=0.25)
    mailbox.update(
        {
            "command": [0.15, 0.0, 0.0],
            "runtime_state": "NAVIGATION",
            "sensor_sequence": 100,
            "sensor_timestamp_monotonic": time.monotonic() - 5.0,
        }
    )

    def publish_fresh_result():
        time.sleep(0.01)
        mailbox.update(
            {
                "command": [0.1, 0.0, 0.05],
                "runtime_state": "NAVIGATION",
                "sensor_sequence": 101,
                "sensor_timestamp_monotonic": time.monotonic(),
            }
        )

    thread = threading.Thread(target=publish_fresh_result)
    thread.start()
    command = mailbox.wait_for_fresh_command(0.2)
    thread.join()
    np.testing.assert_allclose(command.as_array(), [0.1, 0.0, 0.05])


def test_navigation_failed_result_cannot_block_later_fresh_result():
    mailbox = NavigationMailbox(max_age=0.25)
    mailbox.update({"runtime_state": "NAVIGATION_FAILED", "fault_reason": "disconnected"})

    def publish_fresh_result():
        time.sleep(0.01)
        mailbox.update(
            {
                "command": [0.1, 0.0, 0.0],
                "runtime_state": "NAVIGATION",
                "sensor_sequence": 1,
                "sensor_timestamp_monotonic": time.monotonic(),
            }
        )

    thread = threading.Thread(target=publish_fresh_result)
    thread.start()
    command = mailbox.wait_for_fresh_command(0.2)
    thread.join()
    np.testing.assert_allclose(command.as_array(), [0.1, 0.0, 0.0])


def test_reader_thread_alive_status_is_available_after_handoff():
    mailbox = NavigationMailbox(max_age=0.25)
    navigation = NavigationProcess({}, mailbox)
    status = navigation.status()
    assert status["worker_alive"] is False
    assert status["reader_thread_alive"] is False
    assert status["reader_error"] is None
    assert status["reader_exit_reason"] is None
    navigation.stop()


def test_reader_lifetime_survives_two_second_prearm_and_keeps_sequence_growing():
    mailbox = NavigationMailbox(max_age=0.25)
    navigation = NavigationProcess({}, mailbox)

    class FakeWorker:
        def is_alive(self):
            return True

    navigation.process = FakeWorker()
    navigation.receiver_thread = threading.Thread(target=navigation._receive, daemon=True)
    navigation.receiver_thread.start()
    deadline = time.monotonic() + 2.0
    sequence = 0
    while time.monotonic() < deadline:
        navigation.sender.send(
            {
                "command": [0.0, 0.0, 0.0],
                "runtime_state": "NAVIGATION",
                "sensor_sequence": sequence,
                "sensor_timestamp_monotonic": time.monotonic(),
            }
        )
        sequence += 1
        time.sleep(0.02)

    status = navigation.status()
    assert status["worker_alive"] is True
    assert status["reader_thread_alive"] is True
    assert status["reader_error"] is None
    assert status["last_sequence"] >= 90
    assert status["socket_state"] == "unknown"

    navigation.stop_event.set()
    navigation.receiver_thread.join(timeout=1.0)
    navigation.receiver.close()
    navigation.sender.close()


def test_navigation_arm_failure_returns_to_safe_hold_without_active():
    controller = object.__new__(SeaNavHimLocoController)
    controller.state = RuntimeState.ACTIVE
    controller.navigation = SimpleNamespace(
        status=lambda: {
            "worker_alive": False,
            "reader_thread_alive": False,
            "reader_error": "socket closed",
            "reader_exit_reason": "worker_exit",
            "last_sequence": -1,
            "last_packet_age_s": float("inf"),
            "socket_state": "closed",
        }
    )
    handled = SeaNavHimLocoController._handle_policy_arm_failure(
        controller, SafetyError("no fresh navigation command after ACTIVE handoff")
    )
    assert handled is True
    assert controller.state == RuntimeState.DEFAULT_POSE_HOLD
    assert controller.state != RuntimeState.ACTIVE
    assert not SeaNavHimLocoController._handle_policy_arm_failure(
        controller, SafetyError("LowState stale during ACTIVE")
    )


def test_fresh_navigation_command_enters_first_himloco_observation(monkeypatch):
    mailbox = NavigationMailbox(max_age=0.25)
    mailbox.update(
        {
            "command": [0.1, 0.0, 0.05],
            "runtime_state": "NAVIGATION",
            "sensor_sequence": 200,
            "sensor_timestamp_monotonic": time.monotonic(),
        }
    )
    seen = {}

    def fake_initialize(self, snapshot, command):
        seen["snapshot"] = snapshot
        seen["command"] = command
        return object()

    controller = object.__new__(SeaNavHimLocoController)
    controller.navigation = SimpleNamespace(mailbox=mailbox)
    monkeypatch.setattr(
        SeaNavHimLocoController,
        "_hold_default_pose_while_waiting_for_navigation",
        lambda self: None,
    )
    monkeypatch.setattr(FixedHIMLocoController, "_initialize_policy_history", fake_initialize)

    def publish_next():
        time.sleep(0.01)
        mailbox.update(
            {
                "command": [0.08, 0.0, 0.02],
                "runtime_state": "NAVIGATION",
                "sensor_sequence": 201,
                "sensor_timestamp_monotonic": time.monotonic(),
            }
        )

    thread = threading.Thread(target=publish_next)
    thread.start()
    observation = SeaNavHimLocoController._initialize_policy_history(
        controller, object(), FixedCommand(0.0, 0.0, 0.0)
    )
    thread.join()
    assert observation is not None
    np.testing.assert_allclose(seen["command"].as_array(), [0.08, 0.0, 0.02])


def test_goal_tolerance_is_explicit():
    assert goal_is_reached([0.2, 0.2], 0.3)
    assert not goal_is_reached([0.3, 0.1], 0.3)


def test_navigation_requires_the_1460_himloco_hash():
    assert HIM_1460_SHA256 == "cab2489dda7732a7d6f51595aa6362384445c738537d0c1c91569054c7b9f5d1"


def test_navigation_default_rejects_policy_1():
    with pytest.raises(SafetyError, match="approved HIMLoco 1460"):
        validate_navigation_himloco_policy(
            "models/locomotion/himloco/policy_1.pt",
            allow_override=False,
        )


def test_navigation_override_accepts_policy_1_and_keeps_contract_gate():
    path = "models/locomotion/himloco/policy_1.pt"
    sha = validate_navigation_himloco_policy(path, allow_override=True)
    assert sha == "456218effd3a4befdbcd54f85c4474c1aa282df1dd81894bd7761543c18dd11a"

    controller = object.__new__(FixedHIMLocoController)
    controller.args = SimpleNamespace(policy=path, allow_policy_override=True)
    controller.validate_model()
    assert controller.policy_profile == "legacy_policy_1"
    assert controller.policy is not None


def test_navigation_policy_override_accepts_first_observation(monkeypatch):
    seen = {}

    def fake_run_policy_once(self, command, snapshot, observation=None):
        seen["command"] = command
        seen["snapshot"] = snapshot
        seen["observation"] = observation

    mailbox = NavigationMailbox(max_age=1.0)
    mailbox.update(
        {
            "command": [0.12, 0.0, -0.08],
            "runtime_state": "NAVIGATION",
            "sensor_sequence": 1,
            "sensor_timestamp_monotonic": time.monotonic(),
        }
    )
    controller = object.__new__(SeaNavHimLocoController)
    controller.navigation = SimpleNamespace(mailbox=mailbox)
    controller._last_navigation_fault = None
    monkeypatch.setattr(FixedHIMLocoController, "_run_policy_once", fake_run_policy_once)

    observation = object()
    snapshot = object()
    SeaNavHimLocoController._run_policy_once(
        controller,
        FixedCommand(0.0, 0.0, 0.0),
        snapshot,
        observation=observation,
    )

    np.testing.assert_allclose(seen["command"].as_array(), [0.12, 0.0, -0.08])
    assert seen["snapshot"] is snapshot
    assert seen["observation"] is observation

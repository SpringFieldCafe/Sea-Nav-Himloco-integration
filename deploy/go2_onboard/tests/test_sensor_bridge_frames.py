import pytest

from deploy.go2_onboard.goal import Goal2D, GoalManager
from deploy.go2_onboard.ros_state_reader import OdomFrameStats, is_fresh
from deploy.go2_onboard.sensor_bridge import odom_readiness, resolve_odom_frame


def test_odom_frame_uses_callback_data_when_dataclass_frame_is_empty():
    frame, used_fallback = resolve_odom_frame("", "odom")
    assert frame == "odom"
    assert used_fallback is False


def test_empty_odom_frame_is_strict_without_explicit_fallback():
    frame, used_fallback = resolve_odom_frame("", "")
    assert frame == ""
    assert used_fallback is False

    manager = GoalManager()
    manager.update(Goal2D("odom", 1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="does not match"):
        manager.relative_xy([0.0, 0.0], 0.0, frame)


def test_empty_odom_frame_uses_only_explicit_fallback():
    frame, used_fallback = resolve_odom_frame("", "", "odom")
    assert frame == "odom"
    assert used_fallback is True

    manager = GoalManager()
    manager.update(Goal2D("odom", 1.0, 0.0, 1.0))
    assert manager.relative_xy([0.0, 0.0], 0.0, frame).tolist() == [1.0, 0.0]


def test_nonempty_wrong_odom_frame_is_never_overridden_by_fallback():
    frame, used_fallback = resolve_odom_frame("map", "", "odom")
    assert frame == "map"
    assert used_fallback is False

    manager = GoalManager()
    manager.update(Goal2D("odom", 1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="does not match"):
        manager.relative_xy([0.0, 0.0], 0.0, frame)


def test_odom_frame_gate_accepts_only_odom_base_link():
    stats = OdomFrameStats()

    assert stats.accept("odom", "base_link") is True
    assert stats.accept("", "") is False
    assert stats.accept("map", "base_link") is False
    assert stats.accept("odom", "") is False

    assert stats.total_messages == 4
    assert stats.valid_frame_messages == 1
    assert stats.empty_frame_messages == 1
    assert stats.wrong_frame_messages == 2
    assert stats.wrong_child_frame_messages == 1


def test_invalid_odom_frames_do_not_replace_last_valid_sample_or_freshness():
    stats = OdomFrameStats()
    latest_valid_received_at = 10.0

    assert stats.accept("odom", "base_link") is True
    # The caller updates this timestamp only for accepted samples.
    assert stats.accept("", "") is False
    assert stats.accept("map", "base_link") is False
    assert latest_valid_received_at == 10.0
    assert is_fresh(0.09, 0.10) is True
    assert is_fresh(0.11, 0.10) is False


def test_interleaved_odom_frames_keep_valid_stream_count():
    stats = OdomFrameStats()
    frames = [
        ("odom", "base_link"),
        ("", ""),
        ("odom", "base_link"),
        ("map", "base_link"),
        ("odom", "base_link"),
    ]

    accepted = [stats.accept(frame, child) for frame, child in frames]
    assert accepted == [True, False, True, False, True]
    assert stats.valid_frame_messages == 3
    assert stats.empty_frame_messages == 1
    assert stats.wrong_frame_messages == 1


def test_odom_readiness_waits_without_a_valid_sample():
    ready, state = odom_readiness(None, "", "", None, 0.10)
    assert ready is False
    assert state == "WAITING_FOR_ODOM"


def test_odom_readiness_marks_old_valid_sample_stale():
    ready, state = odom_readiness(object(), "odom", "base_link", 0.11, 0.10)
    assert ready is False
    assert state == "SENSOR_STALE"


def test_odom_readiness_recovers_when_valid_sample_is_fresh():
    ready, state = odom_readiness(object(), "odom", "base_link", 0.01, 0.10)
    assert ready is True
    assert state == "RUNNING"

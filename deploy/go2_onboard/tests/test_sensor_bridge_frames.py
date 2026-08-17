import pytest

from deploy.go2_onboard.goal import Goal2D, GoalManager
from deploy.go2_onboard.sensor_bridge import resolve_odom_frame


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

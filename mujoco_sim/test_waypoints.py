import numpy as np

from .waypoints import Waypoint, WaypointManager


def test_waypoint_order_and_final_stop():
    manager = WaypointManager([Waypoint(1.0, 0.0, 0.2), Waypoint(2.0, 0.0, 0.3)])
    assert manager.current_index == 0
    assert not manager.update([0.7, 0.0])
    assert manager.update([1.0, 0.0])
    assert manager.current_index == 1
    assert not manager.done
    assert manager.update([2.0, 0.0])
    assert manager.done
    assert manager.reached_indices == [0, 1]


def test_relative_goal_uses_body_rotation():
    manager = WaypointManager.single(0.0, 1.0, 0.5)
    rotation_world_from_body = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    relative = manager.relative_goal([0.0, 0.0], rotation_world_from_body)
    np.testing.assert_allclose(relative, [1.0, 0.0])

import numpy as np
import pytest


def test_waypoint_visualizer_uses_public_user_scene_geometry():
    mujoco = pytest.importorskip("mujoco")
    from .visualization import WaypointVisualizer
    from .waypoints import Waypoint

    model = mujoco.MjModel.from_xml_path("mujoco_sim/assets/go2/course_mixed.xml")

    class ViewerStub:
        user_scn = mujoco.MjvScene(model, 32)

    viewer = ViewerStub()
    WaypointVisualizer(viewer, [Waypoint(1.0, 0.0), Waypoint(2.0, 0.0)]).sync(
        0, 1.0, "NAVIGATE", np.ones(3), np.zeros(3), False)
    assert viewer.user_scn.ngeom == 3
    assert viewer.user_scn.geoms[2].label

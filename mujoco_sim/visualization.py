import numpy as np


class WaypointVisualizer:
    """Draw waypoint markers and compact runtime diagnostics in a passive viewer."""

    def __init__(self, viewer, waypoints):
        import mujoco
        self.mujoco = mujoco
        self.viewer = viewer
        self.waypoints = waypoints

    def sync(self, current_index, distance, supervisor_state, raw_command,
             supervised_command, goal_reached):
        scene = self.viewer.user_scn
        scene.ngeom = 0
        identity = np.eye(3, dtype=np.float64).ravel()
        for index, waypoint in enumerate(self.waypoints):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            self.mujoco.mjv_initGeom(
                geom, self.mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.13, 0.13, 0.13]),
                np.array([waypoint.x, waypoint.y, 0.18]), identity,
                np.array([0.15, 0.85, 0.2, 0.95] if index < current_index else
                          ([0.9, 0.2, 0.15, 1.0] if index == current_index else
                           [0.2, 0.45, 0.95, 0.7])))
            scene.ngeom += 1
        text = (
            f"index={current_index + 1}/{len(self.waypoints)}  "
            f"distance={distance:.2f} m\n"
            f"state={supervisor_state} reached={goal_reached}\n"
            f"raw=[{raw_command[0]:+.2f}, {raw_command[1]:+.2f}, {raw_command[2]:+.2f}]\n"
            f"supervised=[{supervised_command[0]:+.2f}, {supervised_command[1]:+.2f}, {supervised_command[2]:+.2f}]"
        )
        if scene.ngeom < scene.maxgeom:
            label_geom = scene.geoms[scene.ngeom]
            current = self.waypoints[min(current_index, len(self.waypoints) - 1)]
            self.mujoco.mjv_initGeom(
                label_geom, self.mujoco.mjtGeom.mjGEOM_LABEL,
                np.zeros(3), np.array([current.x, current.y, 0.55]), identity,
                np.array([1.0, 1.0, 1.0, 1.0]))
            label_geom.label = text[:99]
            scene.ngeom += 1

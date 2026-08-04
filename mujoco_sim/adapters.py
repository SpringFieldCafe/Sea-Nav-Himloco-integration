import math
import time

import numpy as np
import torch

from deploy.go2_onboard.joint_mapping import make_motor_to_policy
from deploy.go2_onboard.lidar_ray_adapter import LidarRayAdapter

from .state import UnifiedState


def _gravity_and_rpy(quat):
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    gravity = np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                        1 - 2 * (w * w + z * z)], dtype=np.float32)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return gravity, roll, pitch


class StateAdapter:
    """Interface implemented by ROS2 and MuJoCo state sources."""

    def read(self, goal_xy):
        raise NotImplementedError


class MuJoCoStateAdapter(StateAdapter):
    """Read named Go2 state and a 41-ray body-frame lidar from MuJoCo."""

    def __init__(self, model, data, goal_xy, terrain_hint="flat", ray_mode="grid2ray"):
        self.model, self.data = model, data
        self.goal_xy = np.asarray(goal_xy, dtype=np.float32)
        self.terrain_hint = terrain_hint
        if ray_mode not in ("grid2ray", "physical_lidar"):
            raise ValueError("ray_mode must be grid2ray or physical_lidar")
        self.ray_mode = ray_mode
        self.joint_ids = [model.joint(name).id for name in (
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint")]
        self.ray_adapter = LidarRayAdapter(torch.device("cpu"))

    def _ray(self, origin, direction):
        import mujoco
        geom_id = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model, self.data, np.asarray(origin, dtype=np.float64).reshape(3, 1),
            np.asarray(direction, dtype=np.float64).reshape(3, 1),
            np.ones((6, 1), dtype=np.uint8), 1, self.model.body("base").id, geom_id)
        return float(distance), int(geom_id[0])

    def _physical_rays(self):
        origin = np.asarray(self.data.xpos[self.model.body("base").id]) + np.array([0, 0, .05])
        angles = np.linspace(-2 * math.pi / 3, 2 * math.pi / 3, 41)
        points = []
        for angle in angles:
            vec = np.array([math.cos(angle), math.sin(angle), 0.0])
            distance, geom_id = self._ray(origin, vec)
            if distance >= 0 and self.model.geom(geom_id).bodyid == 0:
                points.append(origin + vec * min(float(distance), 5.0))
        if not points:
            return np.full(41, 5.0, dtype=np.float32)
        # Convert world hit points to body frame before reusing the deployed projector.
        body_id = self.model.body("base").id
        rot = np.asarray(self.data.xmat[body_id]).reshape(3, 3)
        body_points = (np.asarray(points) - self.data.xpos[body_id]) @ rot
        return self.ray_adapter.project(body_points).numpy()[0]

    def _grid2ray(self):
        """Approximate Isaac Gym grid2ray using MuJoCo terrain truth.

        For each horizontal sample, cast a vertical ray and report the first
        sample whose surface is at least 0.1 m above the local ground plane.
        This preserves the trained 41-distance interface without changing the
        SEA-Nav observation ordering.
        """
        body_id = self.model.body("base").id
        body_pos = np.asarray(self.data.xpos[body_id])
        rot = np.asarray(self.data.xmat[body_id]).reshape(3, 3)
        ground_origin = body_pos + np.array([0, 0, 2.0])
        ground_distance, _ = self._ray(ground_origin, [0, 0, -1])
        ground_z = ground_origin[2] - ground_distance if ground_distance >= 0 else 0.0
        values = np.full(41, 5.0, dtype=np.float32)
        angles = np.linspace(-2 * math.pi / 3, 2 * math.pi / 3, 41)
        for i, angle in enumerate(angles):
            direction = rot @ np.array([math.cos(angle), math.sin(angle), 0.0])
            for radius in np.arange(0.1, 5.01, 0.1):
                sample = body_pos + direction * radius + np.array([0, 0, 2.0])
                distance, geom_id = self._ray(sample, [0, 0, -1])
                if distance < 0:
                    continue
                surface_z = sample[2] - distance
                if surface_z > ground_z + 0.1 and self.model.geom(geom_id).bodyid == 0:
                    values[i] = radius
                    break
        return values

    def _rays(self):
        return self._grid2ray() if self.ray_mode == "grid2ray" else self._physical_rays()

    def _obstacle_collision(self):
        robot_ids = {i for i in range(self.model.ngeom) if self.model.geom(i).bodyid != 0}
        terrain_prefixes = ("floor", "step", "stair", "rough_field", "hfield")
        for contact in self.data.contact[:self.data.ncon]:
            first, second = contact.geom1, contact.geom2
            if first in robot_ids and second not in robot_ids:
                other = self.model.geom(second).name or ""
                if other and not other.startswith(terrain_prefixes):
                    return True
            if second in robot_ids and first not in robot_ids:
                other = self.model.geom(first).name or ""
                if other and not other.startswith(terrain_prefixes):
                    return True
        return False

    def read(self, goal_xy=None):
        body_id = self.model.body("base").id
        quat = self.data.xquat[body_id]
        gravity, roll, pitch = _gravity_and_rpy(quat)
        rot = np.asarray(self.data.xmat[body_id]).reshape(3, 3)
        lin = rot.T @ self.data.cvel[body_id, 3:6]
        ang = rot.T @ self.data.cvel[body_id, :3]
        q = np.asarray([self.data.qpos[self.model.jnt_qposadr[j]] for j in self.joint_ids])
        dq = np.asarray([self.data.qvel[self.model.jnt_dofadr[j]] for j in self.joint_ids])
        position = self.data.xpos[body_id][:2].astype(np.float32)
        target = self.goal_xy if goal_xy is None else np.asarray(goal_xy, dtype=np.float32)
        relative_goal = rot[:2, :2].T @ (target - position)
        rays = self._rays()
        terrain_hint = self.terrain_hint(position[0]) if callable(self.terrain_hint) else self.terrain_hint
        state = UnifiedState(gravity, ang, lin, q.astype(np.float32), dq.astype(np.float32),
                             position, rays, relative_goal.astype(np.float32),
                             roll, pitch, terrain_hint,
                             float(np.min(rays)),
                             self._obstacle_collision(), abs(roll) > 1.0 or abs(pitch) > 1.0,
                             time.monotonic(), {"ray_mode": self.ray_mode})
        state.validate()
        return state


class ROS2StateAdapter(StateAdapter):
    """Convert the existing ROS2 reader payload to UnifiedState."""

    def __init__(self, reader, ray_adapter=None):
        self.reader = reader
        self.ray_adapter = ray_adapter or LidarRayAdapter(torch.device("cpu"))
        self.motor_to_policy = make_motor_to_policy()

    def read(self, goal_xy):
        low, odom = self.reader.lowstate.value, self.reader.odom.value
        if low is None or odom is None or self.reader.lidar.value is None:
            raise RuntimeError("ROS2 state is incomplete")
        gravity, _, _ = _gravity_and_rpy(low.quaternion)
        rays = self.ray_adapter.project(low.lidar if hasattr(low, "lidar") else self.reader.lidar.value).numpy()[0]
        state = UnifiedState(gravity, odom.angular_velocity, odom.linear_velocity,
                             np.asarray(low.q_motor)[self.motor_to_policy],
                             np.asarray(low.dq_motor)[self.motor_to_policy], odom.position[:2],
                             rays, np.asarray(goal_xy, dtype=np.float32))
        state.validate()
        return state

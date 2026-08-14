import math

import numpy as np


class LidarRayAdapter:
    """Project a body-frame point cloud into the 41 SEA-Nav rays.

    The expected body convention is x-forward, y-left, z-up.  This is the
    convention of ``cloud_base``; if the deployed driver uses another frame,
    the transform must be configured before hardware use.
    """

    def __init__(self, device, ray_count=41, min_distance=0.1, max_distance=5.0,
                 angle_min=-2.0 * math.pi / 3.0, angle_max=2.0 * math.pi / 3.0,
                 min_z=-0.25, max_z=1.0):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.ray_count = ray_count
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.min_z = min_z
        self.max_z = max_z
        self.angles = torch.linspace(angle_min, angle_max, ray_count, device=self.device)
        self.bin_width = (angle_max - angle_min) / (ray_count - 1)

    def project(self, points):
        torch = self._torch
        points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"point cloud must have shape [M,3+], got {tuple(points.shape)}")
        points = points[:, :3]
        distance = torch.linalg.vector_norm(points[:, :2], dim=1)
        valid = torch.isfinite(points).all(dim=1)
        valid &= distance >= self.min_distance
        valid &= distance <= self.max_distance
        valid &= points[:, 2] >= self.min_z
        valid &= points[:, 2] <= self.max_z
        angle = torch.atan2(points[:, 1], points[:, 0])
        valid &= angle >= self.angles[0] - self.bin_width / 2
        valid &= angle <= self.angles[-1] + self.bin_width / 2
        rays = torch.full((self.ray_count,), self.max_distance, device=self.device)
        if valid.any():
            valid_distance = distance[valid]
            valid_angle = angle[valid]
            index = torch.round((valid_angle - self.angles[0]) / self.bin_width).long()
            index = index.clamp(0, self.ray_count - 1)
            for i in range(self.ray_count):
                selected = valid_distance[index == i]
                if selected.numel() > 0:
                    rays[i] = selected.min()
        return rays.unsqueeze(0).clamp(self.min_distance, self.max_distance)


class NumpyLidarRayAdapter:
    """Torch-free 41-ray projection for read-only diagnostics."""

    def __init__(self, ray_count=41, min_distance=0.1, max_distance=5.0,
                 angle_min=-2.0 * math.pi / 3.0, angle_max=2.0 * math.pi / 3.0,
                 min_z=-0.25, max_z=1.0):
        self.ray_count = ray_count
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.min_z = min_z
        self.max_z = max_z
        self.angles = np.linspace(angle_min, angle_max, ray_count, dtype=np.float32)
        self.bin_width = (angle_max - angle_min) / (ray_count - 1)

    def project(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"point cloud must have shape [M,3+], got {points.shape}")
        points = points[:, :3]
        distance = np.linalg.norm(points[:, :2], axis=1)
        valid = np.isfinite(points).all(axis=1)
        valid &= distance >= self.min_distance
        valid &= distance <= self.max_distance
        valid &= points[:, 2] >= self.min_z
        valid &= points[:, 2] <= self.max_z
        angle = np.arctan2(points[:, 1], points[:, 0])
        valid &= angle >= self.angles[0] - self.bin_width / 2
        valid &= angle <= self.angles[-1] + self.bin_width / 2
        rays = np.full((self.ray_count,), self.max_distance, dtype=np.float32)
        if valid.any():
            index = np.rint((angle - self.angles[0]) / self.bin_width).astype(np.int64)
            index = np.clip(index, 0, self.ray_count - 1)
            for i in range(self.ray_count):
                selected = distance[valid & (index == i)]
                if selected.size:
                    rays[i] = selected.min()
        return np.clip(rays, self.min_distance, self.max_distance)[None, :]

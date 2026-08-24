import math
import threading
import time

import numpy as np


# Narrow masks fitted from repeated stationary cloud_base samples.  The
# center cluster and the two side clusters are kept separate so nearby real
# obstacles at x=0.35/0.50 remain visible.
ROBOT_SELF_MASKS = (
    (0.61, 0.65, -0.07, 0.07, -0.26, -0.22),
    (0.36, 0.48, 0.19, 0.22, -0.26, -0.22),
    (0.36, 0.48, -0.22, -0.19, -0.26, -0.22),
)


def is_robot_self_point(points):
    points = np.asarray(points)
    result = np.zeros(points.shape[:-1], dtype=bool)
    for x_min, x_max, y_min, y_max, z_min, z_max in ROBOT_SELF_MASKS:
        result |= ((points[..., 0] >= x_min) & (points[..., 0] <= x_max) &
                   (points[..., 1] >= y_min) & (points[..., 1] <= y_max) &
                   (points[..., 2] >= z_min) & (points[..., 2] <= z_max))
    return result


class LidarRayAdapter:
    """Project a body-frame point cloud into the 41 SEA-Nav rays.

    The expected body convention is x-forward, y-left, z-up.  This is the
    convention of ``cloud_base``; if the deployed driver uses another frame,
    the transform must be configured before hardware use.
    """

    def __init__(self, device, ray_count=41, min_distance=0.1, max_distance=5.0,
                 angle_min=-2.0 * math.pi / 3.0, angle_max=2.0 * math.pi / 3.0,
                 min_z=-0.15, max_z=1.0):
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
        valid &= ~torch.as_tensor(is_robot_self_point(points.detach().cpu().numpy()), device=points.device)
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
                 min_z=-0.15, max_z=1.0):
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
        valid &= ~is_robot_self_point(points)
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


class LidarRayCache:
    """Cache one projected LiDAR frame for the fixed-rate sensor snapshot.

    Point-cloud projection is deliberately performed by ``update`` only.  A
    snapshot consumer may read the same result many times without re-running
    the projection or changing its source timestamp.
    """

    def __init__(self, adapter=None):
        # The native Go2 cloud contains a dense floor return around z=-0.2 m.
        # Keep it out of the learned obstacle rays by using the deployment
        # default adapter, whose lower bound is -0.15 m.
        self.adapter = adapter or NumpyLidarRayAdapter(min_z=-0.15)
        self._lock = threading.RLock()
        self._rays = np.full((41,), self.adapter.max_distance, dtype=np.float32)
        self._received_at = 0.0
        self._source_timestamp = 0.0
        self._processed_count = 0
        self._last_processing_ms = 0.0

    def update(self, points, received_at=None, source_timestamp=0.0):
        start = time.perf_counter()
        rays = np.asarray(self.adapter.project(points), dtype=np.float32).reshape(-1)
        if rays.shape != (41,) or not np.isfinite(rays).all():
            raise ValueError(f"cached LiDAR rays must be finite (41,), got {rays.shape}")
        processing_ms = (time.perf_counter() - start) * 1000.0
        with self._lock:
            self._rays = rays.copy()
            self._received_at = time.monotonic() if received_at is None else float(received_at)
            self._source_timestamp = float(source_timestamp)
            self._processed_count += 1
            self._last_processing_ms = processing_ms

    def snapshot(self):
        with self._lock:
            return {
                "rays": self._rays.copy(),
                "received_at": self._received_at,
                "source_timestamp": self._source_timestamp,
                "processed_count": self._processed_count,
                "processing_ms": self._last_processing_ms,
            }

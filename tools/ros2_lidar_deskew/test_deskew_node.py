import numpy as np

from deskew_node import LidarDeskewNode, quat_from_rotvec, quat_to_rotation


def test_zero_rotation_is_identity():
    assert np.allclose(quat_to_rotation(quat_from_rotvec(np.zeros(3))), np.eye(3))


def test_pose_at_interpolates_translation():
    poses = [(0.0, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)),
             (1.0, np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))]
    _, position = LidarDeskewNode.pose_at(poses, 0.5)
    assert np.allclose(position, [0.5, 0.0, 0.0])

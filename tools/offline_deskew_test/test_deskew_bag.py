import math

import numpy as np

from deskew_bag import Pose, pose_at, quat_from_rotvec, quat_to_rot


def test_pose_at_interpolates_translation():
    poses = [Pose(0.0, np.eye(3), np.zeros(3)), Pose(1.0, np.eye(3), np.array([1.0, 0.0, 0.0]))]
    pose = pose_at(poses, 0.5)
    assert pose.position == pytest.approx([0.5, 0.0, 0.0])


def test_quaternion_rotation_vector_is_identity_at_zero():
    assert np.allclose(quat_to_rot(quat_from_rotvec(np.zeros(3))), np.eye(3))


import pytest

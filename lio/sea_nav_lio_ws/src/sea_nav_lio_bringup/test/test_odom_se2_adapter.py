import importlib.util
import math

import numpy as np


MODULE_PATH = "lio/sea_nav_lio_ws/src/sea_nav_lio_bringup/sea_nav_lio_bringup/odom_se2_adapter.py"
SPEC = importlib.util.spec_from_file_location("odom_se2_adapter", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def assert_quaternion_close(actual, expected, atol=1e-12):
    # q and -q represent the same rotation.
    if np.dot(actual, expected) < 0.0:
        expected = tuple(-value for value in expected)
    np.testing.assert_allclose(actual, expected, atol=atol)


def test_native_fixed_transform_is_derived_and_nonplanar():
    rotation, translation = MODULE.unitree_imu_to_base_transform()
    assert abs(translation[2]) > 0.04
    assert abs(rotation[0]) > 0.9 or abs(rotation[1]) > 0.9


def test_identity_pose_preserves_full_fixed_se3():
    fixed = MODULE.unitree_imu_to_base_transform()
    position, orientation = MODULE.transform_pose(
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), fixed
    )
    np.testing.assert_allclose(position, fixed[1], atol=1e-12)
    assert_quaternion_close(orientation, fixed[0])


def test_pure_translation_is_composed_in_sensor_frame():
    fixed = MODULE.unitree_imu_to_base_transform()
    position, orientation = MODULE.transform_pose(
        (1.0, -2.0, 3.0), (0.0, 0.0, 0.0, 1.0), fixed
    )
    expected = tuple((1.0, -2.0, 3.0)[i] + fixed[1][i] for i in range(3))
    np.testing.assert_allclose(position, expected, atol=1e-12)
    assert_quaternion_close(orientation, fixed[0])


def test_roll_pitch_sensor_pose_is_not_discarded():
    fixed = MODULE.unitree_imu_to_base_transform()
    sensor_orientation = MODULE.quaternion_from_rpy(0.2, -0.15, 0.3)
    position, orientation = MODULE.transform_pose(
        (1.0, 2.0, 3.0), sensor_orientation, fixed
    )
    expected_position = tuple(
        (1.0, 2.0, 3.0)[i]
        + MODULE._quat_rotate(sensor_orientation, fixed[1])[i]
        for i in range(3)
    )
    expected_orientation = MODULE._quat_multiply(sensor_orientation, fixed[0])
    np.testing.assert_allclose(position, expected_position, atol=1e-12)
    assert_quaternion_close(orientation, expected_orientation)
    assert abs(orientation[0]) > 1e-3 or abs(orientation[1]) > 1e-3


def test_yaw_pose_rotates_fixed_lever_arm():
    fixed = MODULE.unitree_imu_to_base_transform()
    sensor_orientation = MODULE.quaternion_from_rpy(0.0, 0.0, math.pi / 2.0)
    position, _ = MODULE.transform_pose(
        (0.0, 0.0, 0.0), sensor_orientation, fixed
    )
    np.testing.assert_allclose(
        position, MODULE._quat_rotate(sensor_orientation, fixed[1]), atol=1e-12
    )


def test_twist_includes_angular_lever_arm_contribution():
    fixed = MODULE.unitree_imu_to_base_transform()
    linear, angular = MODULE.transform_twist(
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), fixed
    )
    expected_linear = MODULE._quat_rotate(
        MODULE._quat_conjugate(fixed[0]),
        MODULE._cross((0.0, 0.0, 1.0), fixed[1]),
    )
    expected_angular = MODULE._quat_rotate(
        MODULE._quat_conjugate(fixed[0]), (0.0, 0.0, 1.0)
    )
    np.testing.assert_allclose(linear, expected_linear, atol=1e-12)
    np.testing.assert_allclose(angular, expected_angular, atol=1e-12)


def test_legacy_se2_helpers_and_yaml_interface_remain_compatible(tmp_path):
    base_to_lio = (0.161107686458, 0.000775822090016, -0.0918558091865)
    lio_to_base = MODULE.invert_se2(base_to_lio)
    c, s = math.cos(base_to_lio[2]), math.sin(base_to_lio[2])
    np.testing.assert_allclose(
        lio_to_base,
        (-c * base_to_lio[0] - s * base_to_lio[1],
         s * base_to_lio[0] - c * base_to_lio[1],
         -base_to_lio[2]),
    )

    path = tmp_path / "calibration.yaml"
    path.write_text(
        "base_to_lio_x_m: 0.1\n"
        "base_to_lio_y_m: -0.2\n"
        "base_to_lio_yaw_rad: 0.3\n",
        encoding="utf-8",
    )
    assert MODULE.load_base_to_lio(path) == (0.1, -0.2, 0.3)

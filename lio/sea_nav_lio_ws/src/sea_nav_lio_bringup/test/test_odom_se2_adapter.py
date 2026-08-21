import importlib.util
import math

import numpy as np


MODULE_PATH = "lio/sea_nav_lio_ws/src/sea_nav_lio_bringup/sea_nav_lio_bringup/odom_se2_adapter.py"
SPEC = importlib.util.spec_from_file_location("odom_se2_adapter", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_inverse_and_pose_composition_direction():
    base_to_lio = (0.161107686458, 0.000775822090016, -0.0918558091865)
    lio_to_base = MODULE.invert_se2(base_to_lio)

    c, s = math.cos(base_to_lio[2]), math.sin(base_to_lio[2])
    np.testing.assert_allclose(
        lio_to_base,
        (-c * base_to_lio[0] - s * base_to_lio[1],
         s * base_to_lio[0] - c * base_to_lio[1],
         -base_to_lio[2]),
    )

    original_pose = (1.2, -0.4, 0.7)
    lio_pose = MODULE.compose_lio_base(*original_pose, base_to_lio)
    round_trip = MODULE.compose_lio_base(*lio_pose, lio_to_base)
    np.testing.assert_allclose(round_trip, original_pose, atol=1e-12)

    x, y, yaw = MODULE.compose_lio_base(1.0, 2.0, 0.4, lio_to_base)
    c, s = math.cos(0.4), math.sin(0.4)
    expected_x = 1.0 + c * lio_to_base[0] - s * lio_to_base[1]
    expected_y = 2.0 + s * lio_to_base[0] + c * lio_to_base[1]
    np.testing.assert_allclose((x, y, yaw), (expected_x, expected_y, 0.4 + lio_to_base[2]))


def test_yaml_reads_base_to_lio_without_hardcoding(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text(
        "base_to_lio_x_m: 0.1\n"
        "base_to_lio_y_m: -0.2\n"
        "base_to_lio_yaw_rad: 0.3\n",
        encoding="utf-8",
    )
    assert MODULE.load_base_to_lio(path) == (0.1, -0.2, 0.3)

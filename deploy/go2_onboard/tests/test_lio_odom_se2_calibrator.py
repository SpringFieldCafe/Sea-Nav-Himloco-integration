import math
from pathlib import Path

import numpy as np

from tools.calibrate_lio_odom_se2 import (
    inverse_se2,
    load_calibration_yaml,
    planar_motion_pairs,
    pose_matrix,
    require_validation_motion,
    solve_se2,
    transform_from_se2,
)


def _pose(x, y, yaw):
    return {"x": x, "y": y, "z": 0.0, "yaw": yaw}


def test_se2_hand_eye_recovers_relative_extrinsic():
    solved_true = transform_from_se2((0.12, -0.07, 0.08))
    records = []
    for index in range(80):
        yaw = 0.015 * index + 0.06 * math.sin(index / 5.0)
        lio = pose_matrix(_pose(0.04 * index, 0.3 * math.sin(index / 8.0), yaw))
        native = lio.dot(inverse_se2(solved_true))
        records.append({
            "native": _pose(native[0, 2], native[1, 2], math.atan2(native[1, 0], native[0, 0])),
            "lio": _pose(lio[0, 2], lio[1, 2], math.atan2(lio[1, 0], lio[0, 0])),
        })

    pairs = planar_motion_pairs(records, max_stride=16, min_translation=0.01, min_yaw_rad=math.radians(0.5))
    solved = solve_se2(pairs)
    np.testing.assert_allclose(solved, [0.12, -0.07, 0.08], atol=1e-6)


def test_odom_recorder_has_no_write_path():
    source = Path("tools/record_lio_odom_pairs.py").read_text(encoding="utf-8")
    assert "create_publisher" not in source
    assert "ChannelPublisher" not in source
    assert ".Write(" not in source
    assert "LowCmd" not in source
    assert "ServiceSwitch" not in source


def test_validate_only_uses_fixed_yaml_solution(tmp_path):
    yaml_path = tmp_path / "calibration.yaml"
    yaml_path.write_text(
        "base_to_lio_x_m: 0.12\n"
        "base_to_lio_y_m: -0.07\n"
        "base_to_lio_yaw_rad: 0.08\n",
        encoding="utf-8",
    )
    np.testing.assert_allclose(load_calibration_yaml(yaml_path), [0.12, -0.07, 0.08])
    require_validation_motion({
        "translation_pairs": 1,
        "left_right_signed_yaw_pairs": {"positive": 1, "negative": 1},
    })

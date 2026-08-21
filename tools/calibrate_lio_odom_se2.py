#!/usr/bin/env python3
"""Estimate planar LIO-to-native odometry extrinsics from matched records."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def pose_matrix(pose):
    yaw = float(pose["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    result = np.eye(3, dtype=np.float64)
    result[:2, :2] = ((c, -s), (s, c))
    result[:2, 2] = (float(pose["x"]), float(pose["y"]))
    return result


def relative_motion(first, second):
    return np.linalg.inv(pose_matrix(first)).dot(pose_matrix(second))


def planar_motion_pairs(records, max_stride=20, min_translation=0.03, min_yaw_rad=math.radians(2.0)):
    pairs = []
    strides = [1, 2, 4, 8, 16]
    strides = [stride for stride in strides if stride <= max_stride]
    for index, record in enumerate(records):
        for stride in strides:
            other_index = index + stride
            if other_index >= len(records):
                continue
            native = relative_motion(records[index]["native"], records[other_index]["native"])
            lio = relative_motion(records[index]["lio"], records[other_index]["lio"])
            native_translation = float(np.linalg.norm(native[:2, 2]))
            lio_translation = float(np.linalg.norm(lio[:2, 2]))
            native_yaw = math.atan2(native[1, 0], native[0, 0])
            lio_yaw = math.atan2(lio[1, 0], lio[0, 0])
            if max(native_translation, lio_translation) < min_translation and \
                    max(abs(native_yaw), abs(lio_yaw)) < min_yaw_rad:
                continue
            pairs.append({
                "native": native,
                "lio": lio,
                "native_translation_m": native_translation,
                "lio_translation_m": lio_translation,
                "native_yaw_rad": native_yaw,
                "lio_yaw_rad": lio_yaw,
            })
    return pairs


def solve_se2(pairs):
    if len(pairs) < 3:
        raise ValueError("at least three non-degenerate relative motion pairs are required")
    # Relative planar rotations commute, so yaw cannot be recovered from
    # yaw(A)-yaw(B) alone.  Estimate yaw and translation jointly from the
    # full equation, using several initial yaw hypotheses.
    def residual_and_jacobian(parameters):
        x, y, yaw = parameters
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
        d_rotation = np.asarray(((-s, -c), (c, -s)), dtype=np.float64)
        residuals = []
        jacobians = []
        for pair in pairs:
            native = pair["native"]
            lio = pair["lio"]
            translation = np.asarray((x, y), dtype=np.float64)
            residual = (native[:2, :2] - np.eye(2)).dot(translation)
            residual += native[:2, 2] - rotation.dot(lio[:2, 2])
            jacobian = np.column_stack((
                native[:2, :2] - np.eye(2),
                -d_rotation.dot(lio[:2, 2]),
            ))
            residuals.append(residual)
            jacobians.append(jacobian)
        return np.concatenate(residuals), np.vstack(jacobians)

    best = None
    for initial_yaw in np.linspace(-math.pi, math.pi, 24, endpoint=False):
        parameters = np.asarray((0.0, 0.0, initial_yaw), dtype=np.float64)
        for _ in range(40):
            residual, jacobian = residual_and_jacobian(parameters)
            step, _, _, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            parameters += step
            parameters[2] = wrap_angle(parameters[2])
            if float(np.linalg.norm(step)) < 1e-11:
                break
        error = float(np.dot(*([residual_and_jacobian(parameters)[0]] * 2)))
        if best is None or error < best[0]:
            best = (error, parameters.copy())
    if best is None or not np.isfinite(best[1]).all():
        raise ValueError("SE(2) hand-eye solve was ill-conditioned")
    return best[1]


def transform_from_se2(parameters):
    x, y, yaw = [float(value) for value in parameters]
    c, s = math.cos(yaw), math.sin(yaw)
    result = np.eye(3, dtype=np.float64)
    result[:2, :2] = ((c, -s), (s, c))
    result[:2, 2] = (x, y)
    return result


def inverse_se2(transform):
    result = np.eye(3, dtype=np.float64)
    rotation = transform[:2, :2]
    result[:2, :2] = rotation.T
    result[:2, 2] = -rotation.T.dot(transform[:2, 2])
    return result


def residuals(pairs, solved):
    transform = transform_from_se2(solved)
    position = []
    yaw = []
    for pair in pairs:
        left = pair["native"].dot(transform)
        right = transform.dot(pair["lio"])
        position.append(float(np.linalg.norm(left[:2, 2] - right[:2, 2])))
        left_yaw = math.atan2(left[1, 0], left[0, 0])
        right_yaw = math.atan2(right[1, 0], right[0, 0])
        yaw.append(abs(math.degrees(wrap_angle(left_yaw - right_yaw))))
    return {
        "position_m": summarize(position),
        "yaw_deg": summarize(yaw),
    }


def summarize(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def motion_coverage(pairs):
    return {
        "pair_count": len(pairs),
        "translation_pairs": sum(pair["native_translation_m"] >= 0.03 for pair in pairs),
        "rotation_pairs": sum(abs(pair["native_yaw_rad"]) >= math.radians(2.0) for pair in pairs),
        "left_right_signed_yaw_pairs": {
            "positive": sum(pair["native_yaw_rad"] > math.radians(2.0) for pair in pairs),
            "negative": sum(pair["native_yaw_rad"] < -math.radians(2.0) for pair in pairs),
        },
    }


def yaml_text(result):
    solved = result["solution"]
    inverse = result["solution_inverse"]
    lines = [
        "method: planar_se2_hand_eye",
        "equation: 'A_ij X = X B_ij'",
        "source_A: /utlidar/robot_odom (base_link)",
        "source_B: /sea_nav/lio/odom (LIO reference)",
        "solution_X_direction: base_to_lio",
        f"base_to_lio_x_m: {solved[0]:.12g}",
        f"base_to_lio_y_m: {solved[1]:.12g}",
        f"base_to_lio_yaw_rad: {solved[2]:.12g}",
        "adapter_transform_direction: lio_to_base",
        f"lio_to_base_x_m: {inverse[0]:.12g}",
        f"lio_to_base_y_m: {inverse[1]:.12g}",
        f"lio_to_base_yaw_rad: {inverse[2]:.12g}",
    ]
    return "\n".join(lines) + "\n"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-yaml")
    parser.add_argument("--calibration-yaml")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--max-stride", type=int, default=20)
    parser.add_argument("--min-translation-m", type=float, default=0.03)
    parser.add_argument("--min-yaw-deg", type=float, default=2.0)
    return parser


def load_records(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", payload)
    if not isinstance(records, list):
        raise ValueError("input must contain a records list")
    return records


def load_calibration_yaml(path):
    """Read the scalar YAML emitted by this tool without requiring PyYAML."""
    values = {}
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        try:
            values[key.strip()] = float(raw_value.strip())
        except ValueError:
            continue
    keys = ("base_to_lio_x_m", "base_to_lio_y_m", "base_to_lio_yaw_rad")
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"calibration YAML missing keys: {', '.join(missing)}")
    return np.asarray([values[key] for key in keys], dtype=np.float64)


def require_validation_motion(coverage):
    if coverage["translation_pairs"] == 0:
        raise ValueError("validation trajectory has no qualifying translation")
    signed = coverage["left_right_signed_yaw_pairs"]
    if signed["positive"] == 0 or signed["negative"] == 0:
        raise ValueError("validation trajectory must contain both rotation directions")


def run(args):
    if not args.validate_only and not 0.1 <= args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be in [0.1, 0.5)")
    records = load_records(args.input)
    if len(records) < 20:
        raise ValueError("at least 20 synchronized records are required")

    if args.validate_only:
        solution = load_calibration_yaml(args.calibration_yaml)
        pairs = planar_motion_pairs(
            records, args.max_stride, args.min_translation_m,
            math.radians(args.min_yaw_deg),
        )
        coverage = motion_coverage(pairs)
        require_validation_motion(coverage)
        inverse = inverse_se2(transform_from_se2(solution))
        inverse_parameters = np.asarray((
            inverse[0, 2], inverse[1, 2], math.atan2(inverse[1, 0], inverse[0, 0])
        ))
        result = {
            "validate_only": True,
            "calibration_yaml": str(Path(args.calibration_yaml).expanduser()),
            "input_records": len(records),
            "solution": solution.tolist(),
            "solution_inverse": inverse_parameters.tolist(),
            "validation_motion_coverage": coverage,
            "validation_residual": residuals(pairs, solution),
        }
        print(json.dumps(result, indent=2))
        return result

    split = int(len(records) * (1.0 - args.validation_fraction))
    train_records, validation_records = records[:split], records[split:]
    train_pairs = planar_motion_pairs(
        train_records, args.max_stride, args.min_translation_m,
        math.radians(args.min_yaw_deg),
    )
    validation_pairs = planar_motion_pairs(
        validation_records, args.max_stride, args.min_translation_m,
        math.radians(args.min_yaw_deg),
    )
    solution = solve_se2(train_pairs)
    inverse = inverse_se2(transform_from_se2(solution))
    inverse_parameters = np.asarray((inverse[0, 2], inverse[1, 2], math.atan2(inverse[1, 0], inverse[0, 0])))
    result = {
        "input_records": len(records),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "solution": solution.tolist(),
        "solution_inverse": inverse_parameters.tolist(),
        "training_motion_coverage": motion_coverage(train_pairs),
        "validation_motion_coverage": motion_coverage(validation_pairs),
        "training_residual": residuals(train_pairs, solution),
        "validation_residual": residuals(validation_pairs, solution),
    }
    output = Path(args.output_yaml).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml_text(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"OUTPUT_YAML={output}")
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.validate_only and not args.calibration_yaml:
            raise ValueError("--calibration-yaml is required with --validate-only")
        if not args.validate_only and not args.output_yaml:
            raise ValueError("--output-yaml is required unless --validate-only is used")
        run(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"calibration failed: {exc}") from exc


if __name__ == "__main__":
    main()

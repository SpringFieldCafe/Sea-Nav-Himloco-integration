"""Align Isaac Gym and MuJoCo HIMLoco JSONL diagnostics by control sample."""

import argparse
import json
from pathlib import Path

import numpy as np


FIELDS = {
    "command": ("command", "command"),
    "observation": ("observation", "observation"),
    "policy_action": ("policy_action", "himloco_action"),
    "target_position": ("target_position", "diagnostics.target_position"),
    "torque": ("torque", "diagnostics.torque"),
    "base_ang_vel": ("base_ang_vel", "diagnostics.actual_angular_velocity"),
    "projected_gravity": ("projected_gravity", "diagnostics.gravity"),
    "dof_pos": ("dof_pos", "diagnostics.joint_position"),
    "dof_vel": ("dof_vel", "diagnostics.joint_velocity"),
}

BLOCKS = {
    "command": (0, 3),
    "base_ang_vel": (3, 6),
    "projected_gravity": (6, 9),
    "dof_pos": (9, 21),
    "dof_vel": (21, 33),
    "previous_action": (33, 45),
}


def get(row, path):
    value = row
    for part in path.split("."):
        if part not in value:
            return None
        value = value[part]
    return value


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac", required=True)
    parser.add_argument("--mujoco", required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()
    isaac = load(args.isaac)
    mujoco = load(args.mujoco)
    count = min(len(isaac), len(mujoco), args.max_samples)
    if count == 0:
        raise RuntimeError("both logs must contain at least one record")
    result = {"samples_aligned": count, "fields": {}}
    result["observation_blocks"] = {}
    for name, (start, end) in BLOCKS.items():
        errors = []
        first = None
        for index in range(count):
            left = get(isaac[index], "observation")
            right = get(mujoco[index], "observation")
            if left is None or right is None:
                continue
            left = np.asarray(left[start:end], dtype=np.float64)
            right = np.asarray(right[start:end], dtype=np.float64)
            error = float(np.mean(np.abs(left - right)))
            errors.append(error)
            if first is None and error > 0.1:
                first = {"sample": index, "time_s": float(isaac[index].get("time_s", index * 0.02)), "mae": error}
        if errors:
            result["observation_blocks"][name] = {
                "mean_mae": float(np.mean(errors)),
                "p50_mae": float(np.percentile(errors, 50)),
                "p90_mae": float(np.percentile(errors, 90)),
                "p99_mae": float(np.percentile(errors, 99)),
                "first_threshold_crossing": first,
            }
    for name, (isaac_path, mujoco_path) in FIELDS.items():
        errors = []
        first = None
        for index in range(count):
            left = get(isaac[index], isaac_path)
            right = get(mujoco[index], mujoco_path)
            if left is None or right is None:
                continue
            left = np.asarray(left, dtype=np.float64)
            right = np.asarray(right, dtype=np.float64)
            if left.shape != right.shape:
                continue
            error = float(np.mean(np.abs(left - right)))
            errors.append(error)
            if first is None and error > (0.05 if name in ("command", "projected_gravity") else 0.1):
                first = {"sample": index, "time_s": float(isaac[index].get("time_s", index * 0.02)), "mae": error}
        if errors:
            result["fields"][name] = {
                "mean_mae": float(np.mean(errors)),
                "p50_mae": float(np.percentile(errors, 50)),
                "p90_mae": float(np.percentile(errors, 90)),
                "p99_mae": float(np.percentile(errors, 99)),
                "first_threshold_crossing": first,
            }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

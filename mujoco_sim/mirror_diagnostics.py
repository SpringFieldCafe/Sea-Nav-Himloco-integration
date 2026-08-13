"""Proper sagittal-reflection test on real HIMLoco rollout observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deploy.go2_onboard.model_loader import infer, load_himloco_policy


JOINT_MIRROR = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int64)
HIP_INDICES = np.array([0, 3, 6, 9], dtype=np.int64)


def mirror_joint(values):
    mirrored = np.asarray(values, dtype=np.float32)[..., JOINT_MIRROR].copy()
    mirrored[..., HIP_INDICES] *= -1.0
    return mirrored


def mirror_frame(frame):
    frame = np.asarray(frame, dtype=np.float32).copy()
    frame[1] *= -1.0
    frame[2] *= -1.0
    frame[3] *= -1.0
    frame[5] *= -1.0
    frame[7] *= -1.0
    frame[9:21] = mirror_joint(frame[9:21])
    frame[21:33] = mirror_joint(frame[21:33])
    frame[33:45] = mirror_joint(frame[33:45])
    return frame


def mirror_observation(observation):
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (270,):
        raise ValueError(f"expected 270D observation, got {obs.shape}")
    return np.concatenate([mirror_frame(obs[i:i + 45]) for i in range(0, 270, 45)])


def percentile(values, p):
    return float(np.percentile(np.asarray(values), p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--rollout-log", required=True)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    observations = []
    for line in Path(args.rollout_log).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("observation") is not None:
            observations.append(np.asarray(row["observation"], dtype=np.float32))
    observations = observations[:args.max_samples]
    if len(observations) < 100:
        raise RuntimeError(f"need at least 100 real 270D observations, got {len(observations)}")

    policy = load_himloco_policy(args.policy, torch.device(args.device))
    original = torch.from_numpy(np.stack(observations)).to(args.device)
    mirrored = torch.from_numpy(np.stack([mirror_observation(obs) for obs in observations])).to(args.device)
    with torch.inference_mode():
        action = infer(policy, original, 12).cpu().numpy()
        mirrored_policy_action = infer(policy, mirrored, 12).cpu().numpy()
    mirrored_action = mirror_joint(action)
    error = mirrored_policy_action - mirrored_action
    abs_error = np.abs(error)
    sample_mae = abs_error.mean(axis=1)
    sample_l2 = np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(mirrored_action, axis=1), 1e-6)
    result = {
        "samples": len(observations),
        "joint_mirror": JOINT_MIRROR.tolist(),
        "hip_indices": HIP_INDICES.tolist(),
        "overall_mae": float(abs_error.mean()),
        "relative_l2_mean": float(sample_l2.mean()),
        "relative_l2_p50": percentile(sample_l2, 50),
        "relative_l2_p90": percentile(sample_l2, 90),
        "relative_l2_p99": percentile(sample_l2, 99),
        "absolute_error_p50": percentile(abs_error, 50),
        "absolute_error_p90": percentile(abs_error, 90),
        "absolute_error_p99": percentile(abs_error, 99),
        "per_joint_mae": abs_error.mean(axis=0).tolist(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

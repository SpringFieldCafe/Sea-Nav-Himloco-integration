"""No-write HIMLoco timing benchmark.

This module deliberately does not import Unitree SDK2, ROS2, or create a
publisher. It measures the same observation/history, TorchScript, and action
post-processing path used by the fixed Go2 controller.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .himloco_fixed_control import (
    ACTION_CLIP,
    DEFAULT_ANGLES,
    EXPECTED_INPUT_DIM,
    EXPECTED_OUTPUT_DIM,
    build_observation,
    build_target_q,
    sha256_file,
)
from .himloco_observation import HIMLocoObservation


def _stats(values):
    values = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def run_benchmark(policy_path, steps, torch_threads=None, torch_interop_threads=None):
    if torch_interop_threads is not None:
        torch.set_num_interop_threads(torch_interop_threads)
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)

    policy_path = Path(policy_path).resolve()
    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    probe = torch.zeros((1, EXPECTED_INPUT_DIM), dtype=torch.float32)
    with torch.inference_mode():
        output = policy(probe)
    if tuple(output.shape) != (1, EXPECTED_OUTPUT_DIM):
        raise RuntimeError(f"expected (1,12), got {tuple(output.shape)}")

    with torch.inference_mode():
        for _ in range(10):
            policy(probe)

    him_obs = HIMLocoObservation(torch.device("cpu"))
    command = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    gyro = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    gravity = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    q = DEFAULT_ANGLES.copy()
    dq = np.zeros(12, dtype=np.float32)
    obs_times = []
    policy_times = []
    action_times = []
    compute_times = []

    for _ in range(steps):
        compute_start = time.perf_counter()
        obs_start = time.perf_counter()
        observation = build_observation(him_obs, command, gyro, gravity, q, dq)
        obs_times.append(time.perf_counter() - obs_start)

        policy_start = time.perf_counter()
        with torch.inference_mode():
            action_tensor = policy(observation)
        policy_times.append(time.perf_counter() - policy_start)

        action_start = time.perf_counter()
        action = action_tensor.reshape(-1).detach().cpu().numpy()
        target = build_target_q(action)
        clipped = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        him_obs.record_action(torch.from_numpy(clipped).reshape(1, 12))
        if target.shape != (12,) or not np.isfinite(target).all():
            raise RuntimeError("non-finite action processing result")
        action_times.append(time.perf_counter() - action_start)
        compute_times.append(time.perf_counter() - compute_start)

    return {
        "policy": str(policy_path),
        "sha256": sha256_file(str(policy_path)),
        "steps": steps,
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "obs_build": _stats(obs_times),
        "policy_forward": _stats(policy_times),
        "action_processing": _stats(action_times),
        "control_compute": _stats(compute_times),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="No-write HIMLoco 1460 timing benchmark")
    parser.add_argument(
        "--policy",
        default="models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, choices=(1, 2, 4), default=None)
    parser.add_argument("--torch-interop-threads", type=int, choices=(1, 2, 4), default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_benchmark(args.policy, args.steps, args.torch_threads, args.torch_interop_threads)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

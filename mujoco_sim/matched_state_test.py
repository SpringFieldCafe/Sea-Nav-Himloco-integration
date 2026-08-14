"""Replay an Isaac Gym matched state/action sequence in MuJoCo."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from deploy.go2_onboard.model_loader import infer, load_himloco_policy


DEFAULT = np.array([.1, .8, -1.5, -.1, .8, -1.5, .1, 1., -1.5, -.1, 1., -1.5], dtype=np.float64)


def snapshot(model, data):
    import mujoco
    base = model.body("base").id
    mujoco.mj_forward(model, data)
    return {
        "root_position": data.qpos[:3].copy().tolist(),
        "root_quaternion_wxyz": data.qpos[3:7].copy().tolist(),
        "root_linear_velocity_world": data.qvel[:3].copy().tolist(),
        "root_angular_velocity_world": data.qvel[3:6].copy().tolist(),
        "joint_position_policy_order": data.qpos[7:19].copy().tolist(),
        "joint_velocity_policy_order": data.qvel[6:18].copy().tolist(),
        "roll": float(np.arctan2(2 * (data.qpos[3] * data.qpos[4] + data.qpos[5] * data.qpos[6]), 1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2))),
        "pitch": float(np.arcsin(np.clip(2 * (data.qpos[3] * data.qpos[5] - data.qpos[6] * data.qpos[4]), -1, 1))),
    }


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--scene", default="himloco_flat")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import mujoco
    from .run import SCENES, _himloco_pd_torque, HIMLOCO_ACTION_CLIP

    export = json.loads(Path(args.export).read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(SCENES[args.scene]))
    data = mujoco.MjData(model)
    initial = export["initial_state"]
    data.qpos[:3] = initial["root_position"]
    xyzw = initial["root_quaternion_xyzw"]
    data.qpos[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    data.qpos[7:19] = initial["joint_position_policy_order"]
    data.qvel[:3] = initial["root_linear_velocity_world"]
    data.qvel[3:6] = initial["root_angular_velocity_world"]
    data.qvel[6:18] = initial["joint_velocity_policy_order"]
    mujoco.mj_forward(model, data)

    policy = load_himloco_policy(args.policy, torch.device(args.device))
    obs = np.asarray(export["observation"], dtype=np.float32)
    with torch.inference_mode():
        replay_action = infer(policy, torch.from_numpy(obs).reshape(1, -1), 12)[0].cpu().numpy()
    saved_action = np.asarray(export["actions"][0], dtype=np.float64)
    print(json.dumps({
        "policy_action_mae": mae(replay_action, saved_action),
        "policy_action_max_abs": float(np.max(np.abs(replay_action - saved_action))),
        "ground": {
            "type": "plane",
            "friction": model.geom_friction[model.geom("floor").id].tolist(),
        },
        "control_contract": {
            "action_scale": 0.25,
            "hip_reduction": 1.0,
            "p_gain": 20.0,
            "d_gain": 0.5,
            "torque_limit": 33.5,
            "joint_order": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        },
        "initial_state": snapshot(model, data),
    }, indent=2))

    control_steps = int(export["decimation"])
    results = []
    for index, action in enumerate(export["actions"]):
        action = np.clip(np.asarray(action, dtype=np.float64), -HIMLOCO_ACTION_CLIP, HIMLOCO_ACTION_CLIP)
        target = DEFAULT + action * .25
        for _ in range(control_steps):
            data.ctrl[:] = _himloco_pd_torque(target, data.qpos[7:19], data.qvel[6:18])
            mujoco.mj_step(model, data)
        current = snapshot(model, data)
        reference = export["post_states"][index]
        if index + 1 in (1, 10, 25, 50):
            results.append({
                "control_intervals": index + 1,
                "time_s": (index + 1) * float(export["control_dt"]),
                "q_mae": mae(current["joint_position_policy_order"], reference["joint_position_policy_order"]),
                "dq_mae": mae(current["joint_velocity_policy_order"], reference["joint_velocity_policy_order"]),
                "base_lin_vel_mae": mae(current["root_linear_velocity_world"], reference["root_linear_velocity_world"]),
                "base_ang_vel_mae": mae(current["root_angular_velocity_world"], reference["root_angular_velocity_world"]),
                "position_mae": mae(current["root_position"], reference["root_position"]),
                "roll_abs_diff": abs(current["roll"] - float(reference.get("roll", 0.0))),
                "pitch_abs_diff": abs(current["pitch"] - float(reference.get("pitch", 0.0))),
            })
    print(json.dumps({"open_loop_divergence": results}, indent=2))


if __name__ == "__main__":
    main()

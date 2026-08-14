# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import sys


from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import time
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import numpy as np
import torch
import time
import cv2
import json
from isaacgym import gymapi


def _roll_pitch_from_quat(quat):
    """Return absolute roll/pitch from Isaac Gym xyzw quaternions."""
    x, y, z, w = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(pitch_arg)
    return roll, pitch


def _isaac_snapshot(env, backend):
    """Serialize the pre-step state in HIMLoco policy joint order."""
    q = env.dof_pos[:, backend.policy_to_sim_device]
    dq = env.dof_vel[:, backend.policy_to_sim_device]
    root = env.root_states[0].detach().cpu().numpy()
    quat_xyzw = root[3:7]
    return {
        "root_position": root[:3].tolist(),
        "root_quaternion_xyzw": quat_xyzw.tolist(),
        "root_linear_velocity_world": root[7:10].tolist(),
        "root_angular_velocity_world": root[10:13].tolist(),
        "joint_position_policy_order": q[0].detach().cpu().numpy().tolist(),
        "joint_velocity_policy_order": dq[0].detach().cpu().numpy().tolist(),
    }

    
def play(args):
    if args.navigation_speed_scale <= 0.0:
        raise ValueError("--navigation_speed_scale must be greater than zero")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # overwrite some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    
    env_cfg.terrain.terrain_types = ['hard_room']  
    env_cfg.terrain.terrain_proportions = [1.0]
    if getattr(args, "matched_flat", False):
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.measure_heights = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        print(
            "[matched-state] Isaac Gym ground=plane, "
            f"static_friction={env_cfg.terrain.static_friction}, "
            f"dynamic_friction={env_cfg.terrain.dynamic_friction}, "
            f"restitution={env_cfg.terrain.restitution}"
        )
    env_cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/go2_description/urdf/go2_description.urdf'
    env_cfg.replay.enable_collision_replay = False
    
    env_cfg.visualization.ray_groups = {
            # "all": [None, "ray_pink"],
            "guidance_navigation": ["guide", "guide_ray_marker"],
        }
    
    if env_cfg.env.num_envs == 1:
        env_cfg.terrain.num_rows = 1 # level  
        env_cfg.terrain.num_cols = 1 # type
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.max_init_terrain_level = 3
    
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [0, 0]
    env_cfg.env.episode_length_s = 40
    env_cfg.env.stay_time = 500
    env_cfg.env.debug_viz = True
    env_cfg.asset.terminate_after_contacts_on = [] # no termination

    fixed_command_test = args.smoke_steps is not None
    if fixed_command_test:
        # Bypass the training command callback for a clean HIMLoco contract
        # test. The command is injected through LeggedRobotPos.step().
        env_cfg.commands.continuous_turning = False
        env_cfg.commands.heading_command = False
        env_cfg.commands.alpha = 1.0
        env_cfg.domain_rand.push_robots = False
        env_cfg.env.goal_reached_time = 1000000
        env_cfg.env.stay_time = 1000000
        env_cfg.env.episode_length_s = max(float(env_cfg.env.episode_length_s), 120.0)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # A fixed-command smoke test exercises HIMLoco directly.  It must not
    # load a SEA-Nav TorchScript export through the PPO checkpoint loader.
    if fixed_command_test:
        policy = None
        print("Running HIMLoco fixed-command smoke test; navigation policy is bypassed.")
    else:
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = -1
        train_cfg.runner.checkpoint = -1
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)
        print('Loaded policy from: ', task_registry.loaded_policy_path)
        print(f"Navigation forward speed scale: {args.navigation_speed_scale:.3f}")

    def navigation_policy(observations):
        with torch.inference_mode():
            actions = policy(observations)
        if actions.ndim != 2 or actions.shape[1] != env.num_nav_actions:
            raise RuntimeError(
                f"SEA-Nav navigation policy must return [N,{env.num_nav_actions}], "
                f"got {tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise FloatingPointError("SEA-Nav navigation policy returned NaN or Inf")
        return actions

    # ---------------------------
    # Camera Setup for Recording
    # ---------------------------
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1000
    camera_props.height = 1000
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    
    # Set camera position (adjust as needed)
    # View from top-down or isometric
    env.gym.set_camera_location(camera_handle, env.envs[0], gymapi.Vec3(5.0, 5.0, 7.0), gymapi.Vec3(4.99, 5.0, 0.0))

    RECORD_VIDEO = False
    SAVE_IMAGES = False
    TOTAL_EPISODES = 10
    video = None
    current_frame = 0
    max_frames = 20000

    env.reset()
    obs, _ = env.reset()
    episode_count = 0

    max_steps = args.smoke_steps if fixed_command_test else 100 * int(env.max_episode_length)
    max_roll = 0.0
    max_pitch = 0.0
    reset_count = 0
    matched_export = None
    matched_actions = []
    matched_posts = []
    contract_handle = open(args.contract_log, "w", encoding="utf-8") if args.contract_log else None
    target_command = torch.tensor(args.smoke_command, device=env.device, dtype=torch.float32).unsqueeze(0)
    target_command = target_command.repeat(env.num_envs, 1)
    pre_roll_command = target_command.clone()
    pre_roll_command[:, 2] = 0.0
    command = pre_roll_command.clone()
    alpha = float(args.smoke_command_filter_alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("--smoke_command_filter_alpha must be in (0, 1]")

    with torch.inference_mode():
        for i in range(max_steps):
            matched_backend = env.locomotion_backend if fixed_command_test else None
            capture_matched = bool(
                fixed_command_test and args.matched_state_export and
                i == int(args.matched_state_step)
            )
            if capture_matched:
                matched_export = {
                    "schema": "matched_himloco_state_v1",
                    "policy_path": os.path.abspath(args.himloco_policy),
                    "step": i,
                    "control_dt": float(env.dt),
                    "decimation": int(env.cfg.control.decimation),
                    "policy_joint_order": list(matched_backend.POLICY_JOINT_NAMES),
                    "initial_state": _isaac_snapshot(env, matched_backend),
                    "observation": None,
                    "previous_action": None,
                    "actions": [],
                    "post_states": [],
                }
            # Step the environment
            if fixed_command_test:
                if i * float(env.dt) < float(args.smoke_pre_roll):
                    desired = pre_roll_command
                else:
                    desired = target_command
                command = alpha * desired + (1.0 - alpha) * command
                actions = command
            else:
                actions = navigation_policy(obs.detach())
                actions[:, 0] *= args.navigation_speed_scale
            obs, _, rews, dones, infos = env.step(actions)
            roll, pitch = _roll_pitch_from_quat(env.base_quat)
            if matched_export is not None and len(matched_actions) < int(args.matched_state_steps):
                backend = env.locomotion_backend
                if matched_export["observation"] is None:
                    matched_export["observation"] = tensor_row(backend.last_observation) if 'tensor_row' in locals() else backend.last_observation[0].detach().cpu().numpy().tolist()
                    matched_export["previous_action"] = backend.last_observation[0, 33:45].detach().cpu().numpy().tolist()
                matched_actions.append(backend.last_policy_action[0].detach().cpu().numpy().tolist())
                matched_posts.append(_isaac_snapshot(env, backend))
                matched_export["actions"] = matched_actions
                matched_export["post_states"] = matched_posts
                if len(matched_actions) == int(args.matched_state_steps):
                    with open(args.matched_state_export, "w", encoding="utf-8") as handle:
                        json.dump(matched_export, handle, indent=2)
                    print(f"Wrote matched Isaac state export: {args.matched_state_export}")
            if contract_handle is not None and fixed_command_test:
                backend = env.locomotion_backend
                def tensor_row(value):
                    if value is None:
                        return None
                    return value[0].detach().cpu().numpy().tolist()
                q_target = env.default_dof_pos[0] + env.actions[0, :12] * float(
                    backend.control_params()["action_scale"]
                )
                row = {
                    "step": i,
                    "time_s": float((i + 1) * env.dt),
                    "command": tensor_row(backend.last_command),
                    "command_scale": list(backend.contract.command_scale),
                    "command_observation": tensor_row(backend.last_scaled_command),
                    "observation": tensor_row(backend.last_observation),
                    "one_step_observation": tensor_row(backend.last_one_step_observation),
                    "policy_action": tensor_row(backend.last_policy_action),
                    "target_position": tensor_row(q_target.unsqueeze(0)),
                    "torque": tensor_row(env.torques),
                    "base_ang_vel": tensor_row(backend.last_raw_angular_velocity),
                    "base_ang_vel_raw": tensor_row(backend.last_raw_angular_velocity),
                    "base_ang_vel_scale": float(backend.contract.angular_velocity_scale),
                    "base_ang_vel_observation": tensor_row(backend.last_scaled_angular_velocity),
                    "projected_gravity": tensor_row(backend.last_raw_gravity),
                    "projected_gravity_raw": tensor_row(backend.last_raw_gravity),
                    "dof_pos": tensor_row(backend.last_raw_dof_position),
                    "dof_pos_raw": tensor_row(backend.last_raw_dof_position),
                    "dof_pos_scale": float(backend.contract.dof_position_scale),
                    "dof_pos_observation": tensor_row(backend.last_scaled_dof_position),
                    "dof_vel": tensor_row(backend.last_raw_dof_velocity),
                    "dof_vel_raw": tensor_row(backend.last_raw_dof_velocity),
                    "dof_vel_scale": float(backend.contract.dof_velocity_scale),
                    "dof_vel_observation": tensor_row(backend.last_scaled_dof_velocity),
                    "contact_forces": tensor_row(env.contact_forces[:, env.feet_indices, :].reshape(env.num_envs, -1)),
                    "roll": float(roll[0].item()),
                    "pitch": float(pitch[0].item()),
                    "done": bool(dones[0].item()),
                    "ground": {
                        "type": "plane" if getattr(args, "matched_flat", False) else "configured",
                        "static_friction": float(env.cfg.terrain.static_friction),
                        "dynamic_friction": float(env.cfg.terrain.dynamic_friction),
                        "restitution": float(env.cfg.terrain.restitution),
                    },
                    "control_contract": {
                        "action_scale": float(backend.contract.action_scale),
                        "hip_reduction": 1.0,
                        "p_gain": float(backend.contract.p_gain),
                        "d_gain": float(backend.contract.d_gain),
                        "torque_limits": env.torque_limits[0].detach().cpu().numpy().tolist(),
                        "joint_order": list(backend.POLICY_JOINT_NAMES),
                    },
                    "reset_reason": {
                        name: bool(values[0].item())
                        for name, values in backend.env.last_reset_reason.items()
                    },
                }
                contract_handle.write(json.dumps(row) + "\n")
                contract_handle.flush()
            max_roll = max(max_roll, float(torch.abs(roll).max().item()))
            max_pitch = max(max_pitch, float(torch.abs(pitch).max().item()))
            reset_count += int(dones.sum().item())
            env.gym.set_camera_location(camera_handle, env.envs[0], gymapi.Vec3(5.0, 5.0, 7.0), gymapi.Vec3(4.99, 5.0, 0.0))

            if dones.any():
                episode_count += 1
                print(f"============== Episode {episode_count} Finished ============== ")

            if episode_count == TOTAL_EPISODES:
                print(f"Reached {TOTAL_EPISODES} episodes, stopping.")
                if video is not None:
                    video.release()  
                break           

            # Recording Logic
            if (RECORD_VIDEO or SAVE_IMAGES) and current_frame < max_frames:
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
                img = img.reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
                
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if RECORD_VIDEO:
                    if video is None:
                        fps = 50
                        output_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', f"{train_cfg.runner.load_run}_{train_cfg.runner.checkpoint}.mp4")
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (camera_props.width, camera_props.height))
                        print(f"Recording video to {output_path}")
                    video.write(img_bgr)

                if SAVE_IMAGES:
                    img_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames')
                    os.makedirs(img_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(img_dir, f"frame_{current_frame:04d}.png"), img_bgr)

                current_frame += 1
                if current_frame % 100 == 0:
                    print(f"Recorded {current_frame}/{max_frames} frames")
            
            elif (RECORD_VIDEO or SAVE_IMAGES) and current_frame >= max_frames:
                if video is not None:
                    video.release()
                    video = None
                RECORD_VIDEO = False
                SAVE_IMAGES = False

    if contract_handle is not None:
        contract_handle.close()

    if fixed_command_test:
        print({
            "mode": "himloco_fixed_command",
            "command": args.smoke_command,
            "steps": args.smoke_steps,
            "max_roll": max_roll,
            "max_pitch": max_pitch,
            "resets": reset_count,
            "policy": os.path.abspath(args.himloco_policy) if args.himloco_policy else None,
        })


if __name__ == '__main__':
    args = get_args()
    args.headless = False
    play(args)

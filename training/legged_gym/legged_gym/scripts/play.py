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
import json
from pathlib import Path

# Prefer the legged_gym package checked out beside this script over an older
# editable install from another worktree.
LOCAL_LEGGED_GYM_ROOT = Path(__file__).resolve().parents[2]
if str(LOCAL_LEGGED_GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEGGED_GYM_ROOT))


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
from isaacgym import gymapi

    
def play(args):
    if args.navigation_speed_scale <= 0.0:
        raise ValueError("--navigation_speed_scale must be greater than zero")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # overwrite some parameters for testing
    if args.viewer:
        env_cfg.env.num_envs = 1
    elif args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    
    env_cfg.terrain.terrain_types = ['hard_room']  
    env_cfg.terrain.terrain_proportions = [1.0]
    env_cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/go2_description/urdf/go2_description.urdf'
    env_cfg.replay.enable_collision_replay = False
    
    env_cfg.visualization.ray_groups = {
            # "all": [None, "ray_pink"],
            "guidance_navigation": ["guide", "guide_ray_marker"],
        }
    
    if args.viewer:
        env_cfg.terrain.num_rows = 1 # level  
        env_cfg.terrain.num_cols = 1 # type
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_init_terrain_level = 0
    
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [0, 0]
    env_cfg.env.episode_length_s = 40
    env_cfg.env.stay_time = 500
    env_cfg.env.debug_viz = True
    # Keep the configured base/head fall termination active for evaluation.

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # The deployment artifact is a critic-free TorchScript actor. Keep the
    # original PPO-checkpoint path available for older training checkpoints.
    navigation_module = None
    navigation_path = os.path.abspath(args.navigation_checkpoint)
    try:
        navigation_module = torch.jit.load(navigation_path, map_location=env.device).eval()
        print('Loaded TorchScript navigation policy from: ', navigation_path, flush=True)
        with torch.inference_mode():
            probe = navigation_module(obs.to(env.device))
        if tuple(probe.shape) != (env.num_envs, env.num_nav_actions):
            raise RuntimeError(
                f"SEA-Nav TorchScript output must be {(env.num_envs, env.num_nav_actions)}, "
                f"got {tuple(probe.shape)}"
            )
    except RuntimeError:
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = -1
        train_cfg.runner.checkpoint = -1
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)
        print('Loaded PPO navigation policy from: ', task_registry.loaded_policy_path, flush=True)
    print(f"Navigation forward speed scale: {args.navigation_speed_scale:.3f}")

    def navigation_policy(observations):
        with torch.inference_mode():
            if navigation_module is not None:
                actions = navigation_module(observations.to(env.device))
            else:
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
    camera_handle = None
    if args.viewer:
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1000
        camera_props.height = 1000
        camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
        env.gym.set_camera_location(
            camera_handle, env.envs[0], gymapi.Vec3(5.0, 5.0, 7.0),
            gymapi.Vec3(4.99, 5.0, 0.0)
        )

    RECORD_VIDEO = False
    SAVE_IMAGES = False
    total_episodes = args.eval_episodes
    if total_episodes <= 0:
        raise ValueError("--eval_episodes must be greater than zero")
    max_steps = args.eval_steps or (10 * int(env.max_episode_length))
    if max_steps <= 0:
        raise ValueError("--eval_steps must be greater than zero")
    if args.eval_log_interval <= 0:
        raise ValueError("--eval_log_interval must be greater than zero")
    eval_handle = None
    if args.eval_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.eval_log)), exist_ok=True)
        eval_handle = open(args.eval_log, "w", encoding="utf-8")
    video = None
    current_frame = 0
    max_frames = 20000

    obs, _ = env.reset()
    episode_count = 0

    try:
        with torch.inference_mode():
            for i in range(max_steps):
                step_start = time.perf_counter()
                position_before = env.root_states[:, :2].detach().clone()
            # Step the environment
                actions = navigation_policy(obs.detach())
                actions[:, 0] *= args.navigation_speed_scale
                raw_actions = actions.detach().clone()
                obs, _, rews, dones, infos = env.step(actions)
                if args.viewer:
                    env.gym.set_camera_location(camera_handle, env.envs[0], gymapi.Vec3(5.0, 5.0, 7.0), gymapi.Vec3(4.99, 5.0, 0.0))

                if eval_handle is not None and i % args.eval_log_interval == 0:
                    env_idx = 0
                    eval_handle.write(json.dumps({
                        "type": "step",
                        "step": i,
                        "env_index": env_idx,
                        "seed": args.seed,
                        "terrain": "hard_room",
                        "position_xy": env.root_states[env_idx, :2].detach().cpu().tolist(),
                        "position_before_xy": position_before[env_idx].cpu().tolist(),
                        "target_xy": env.position_targets[env_idx, :2].detach().cpu().tolist(),
                        "target_relative_xy": env.goal_local_pos[env_idx].detach().cpu().tolist(),
                        "target_distance": float(env.distance[env_idx].item()),
                        "raw_command": raw_actions[env_idx].cpu().tolist(),
                        "filtered_command": env.nav_actions_after_clip[env_idx].detach().cpu().tolist(),
                        "himloco_action": env.actions_orig[env_idx].detach().cpu().tolist(),
                        "ray_min_distance": float(env.rays[env_idx].min().item()),
                        "roll_proxy": float(env.episode_max_roll[env_idx].item()),
                        "pitch_proxy": float(env.episode_max_pitch[env_idx].item()),
                        "control_hz": 1.0 / env.dt,
                        "wall_control_hz": 1.0 / max(time.perf_counter() - step_start, 1e-6),
                    }) + "\n")

                if dones.any():
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    for env_id in done_ids.tolist():
                        episode_count += 1
                        summary = {
                            "type": "episode",
                            "episode": episode_count,
                            "seed": args.seed,
                            "env_index": env_id,
                            "terrain": "hard_room",
                            "start_xy": env.last_episode_start_xy[env_id].detach().cpu().tolist(),
                            "start_yaw": float(env.last_episode_start_yaw[env_id].item()),
                            "target_xy": env.last_episode_goal_xy[env_id].detach().cpu().tolist(),
                            "goal_reached": bool(env.last_episode_goal_reached[env_id].item()),
                            "collision": bool(env.last_episode_collision[env_id].item()),
                            "collision_count": int(env.last_episode_collision_count[env_id].item()),
                            "fallen": bool(env.last_episode_fallen[env_id].item()),
                            "path_length": float(env.last_episode_path_length[env_id].item()),
                            "min_obstacle_distance": float(env.last_episode_min_obstacle_distance[env_id].item()),
                            "max_roll": float(env.last_episode_max_roll[env_id].item()),
                            "max_pitch": float(env.last_episode_max_pitch[env_id].item()),
                            "steps": int(env.last_episode_steps[env_id].item()),
                            "sim_time_s": float(env.last_episode_steps[env_id].item() * env.dt),
                        }
                        if eval_handle is not None:
                            eval_handle.write(json.dumps(summary) + "\n")
                        print(json.dumps(summary, sort_keys=True), flush=True)
                    if eval_handle is not None:
                        eval_handle.flush()

                if episode_count >= total_episodes:
                    print(f"Reached {total_episodes} episodes, stopping.", flush=True)
                    break

            # Video recording remains opt-in and is intentionally separate from
            # the JSONL evaluation path.
    finally:
        if eval_handle is not None:
            eval_handle.close()


if __name__ == '__main__':
    args = get_args()
    play(args)

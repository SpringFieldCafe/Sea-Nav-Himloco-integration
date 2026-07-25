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
from isaacgym import gymapi


def _info_value_to_bool(value):
    if value is None:
        return False
    if torch.is_tensor(value):
        return bool(value.flatten()[0].item()) if value.numel() > 0 else False
    if isinstance(value, np.ndarray):
        return bool(value.flatten()[0]) if value.size > 0 else False
    if isinstance(value, (list, tuple)):
        return bool(value[0]) if len(value) > 0 else False
    return bool(value)


def _info_value_to_int(value, default=0):
    if value is None:
        return default
    if torch.is_tensor(value):
        return int(value.flatten()[0].item()) if value.numel() > 0 else default
    if isinstance(value, np.ndarray):
        return int(value.flatten()[0]) if value.size > 0 else default
    if isinstance(value, (list, tuple)):
        return int(value[0]) if len(value) > 0 else default
    return int(value)


def _get_episode_result(infos):
    reason = infos.get("termination_reason", {})
    if _info_value_to_bool(reason.get("goal_reached")):
        return "success"
    if _info_value_to_bool(reason.get("fall_down")):
        return "fall_down"
    if _info_value_to_bool(reason.get("stand_still")):
        return "stand_still"
    if (
        _info_value_to_bool(reason.get("terminate_contact"))
        or _info_value_to_bool(reason.get("replay_collision_reset"))
        or _info_value_to_bool(reason.get("hard_force_reset"))
    ):
        return "collision"
    if _info_value_to_bool(reason.get("timeout")):
        return "timeout"
    return "other"


def _print_eval_summary(stats):
    total = stats["total"]
    success_rate = 100.0 * stats["success"] / total if total > 0 else 0.0
    mean_length = float(np.mean(stats["episode_lengths"])) if stats["episode_lengths"] else 0.0

    print("================ Evaluation Summary ================")
    print(f"Total episodes: {total}")
    print(f"Successes: {stats['success']}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Collisions: {stats['collision']}")
    print(f"Timeouts: {stats['timeout']}")
    print(f"Stand still: {stats['stand_still']}")
    print(f"Fall down: {stats['fall_down']}")
    print(f"Other terminations: {stats['other']}")
    print(f"Average episode length: {mean_length:.1f} steps")

    
def play(args):
    if args.num_episodes <= 0:
        raise ValueError("--num_episodes must be greater than 0")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # overwrite some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    
    env_cfg.terrain.terrain_types = ['hard_room']  
    env_cfg.terrain.terrain_proportions = [1.0]
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

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    print('Loaded policy from: ', task_registry.loaded_policy_path)

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
    total_episodes = args.num_episodes
    video = None
    current_frame = 0
    max_frames = 20000

    obs, _ = env.reset()
    stats = {
        "total": 0,
        "success": 0,
        "collision": 0,
        "timeout": 0,
        "stand_still": 0,
        "fall_down": 0,
        "other": 0,
        "episode_lengths": [],
    }
    max_eval_steps = total_episodes * (int(env.max_episode_length) + 1)

    with torch.no_grad():
        for i in range(max_eval_steps):
            # Step the environment
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            env.gym.set_camera_location(camera_handle, env.envs[0], gymapi.Vec3(5.0, 5.0, 7.0), gymapi.Vec3(4.99, 5.0, 0.0))

            if dones.any():
                result = _get_episode_result(infos)
                episode_length = _info_value_to_int(infos.get("episode_lengths"), default=0)
                stats["total"] += 1
                stats[result] += 1
                stats["episode_lengths"].append(episode_length)
                print(f"============== Episode {stats['total']} Finished: {result}, length={episode_length} steps ============== ")

            if stats["total"] == total_episodes:
                print(f"Reached {total_episodes} episodes, stopping.")
                if video is not None:
                    video.release()  
                _print_eval_summary(stats)
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

    if stats["total"] < total_episodes:
        print(f"Stopped after {stats['total']} episodes before reaching requested {total_episodes}.")
        if video is not None:
            video.release()
        _print_eval_summary(stats)


if __name__ == '__main__':
    args = get_args()
    args.headless = False
    play(args)

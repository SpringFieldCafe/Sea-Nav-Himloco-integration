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

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch


def _stats(value):
    if value is None:
        return "n/a"
    value = value.detach()
    return f"[{value.min().item():+.3f}, {value.max().item():+.3f}]"


def _print_smoke_summary(env, args, total_steps):
    backend = getattr(env, "locomotion_backend", None)
    print("================ HIMLoco zero-command smoke test ================")
    print(f"task={args.task}")
    print(f"locomotion_backend={getattr(backend, 'name', 'unknown')}")
    print(f"policy={getattr(backend, 'policy_path', 'n/a')}")
    print(f"device={env.device}, num_envs={env.num_envs}, headless={env.headless}")
    print(f"simulation_dt={env.dt:.6f}s, control_frequency={1.0 / env.dt:.2f}Hz")
    print(f"num_actions={env.num_actions}, navigation_command_dim=3")
    print(f"max_episode_length={env.max_episode_length}, total_steps={total_steps}")
    if backend is not None and hasattr(backend, "input_dim"):
        print(
            f"policy_input_dim={backend.input_dim}, "
            f"history_length={backend.history_length}, "
            f"action_scale={backend.contract.action_scale}, "
            f"action_clip={backend.contract.action_clip}"
        )
    print("===================================================================")


def test_env(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs =  min(env_cfg.env.num_envs, 10)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    total_steps = int(10 * env.max_episode_length)
    _print_smoke_summary(env, args, total_steps)
    completed_episodes = 0
    for i in range(total_steps):
        # LeggedRobotPos.step() consumes high-level navigation commands
        # [vx, vy, wz].  The 12-DOF locomotion action is produced internally
        # by the selected backend.
        actions = 0. * torch.ones(env.num_envs, 3, device=env.device)
        obs, _, rew, done, info = env.step(actions)
        if not torch.isfinite(obs).all() or not torch.isfinite(env.torques).all():
            raise FloatingPointError(
                f"non-finite state at step {i}: obs={torch.isfinite(obs).all().item()}, "
                f"torques={torch.isfinite(env.torques).all().item()}"
            )
        completed_episodes += int(done.sum().item())
        if i == 0 or (i + 1) % 100 == 0 or i + 1 == total_steps:
            tilt = torch.linalg.norm(env.projected_gravity[:, :2], dim=1)
            print(
                f"[smoke] step {i + 1}/{total_steps} "
                f"episodes={completed_episodes} "
                f"action={_stats(getattr(env, 'actions_orig', None))} "
                f"torque={_stats(env.torques)} "
                f"tilt_proxy_max={tilt.max().item():.3f}",
                flush=True,
            )

    backend = getattr(env, "locomotion_backend", None)
    print(f"[smoke] completed episodes={completed_episodes}")
    if backend is not None and hasattr(backend, "command_stats"):
        print(f"[smoke] HIM command stats={backend.command_stats()}")
    print("HIMLOCO_ZERO_COMMAND_SMOKE_OK")

if __name__ == '__main__':
    args = get_args()
    test_env(args)

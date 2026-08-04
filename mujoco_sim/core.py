import time

import numpy as np
import torch

from deploy.go2_onboard.himloco_observation import HIMLocoObservation
from deploy.go2_onboard.joint_mapping import make_policy_to_motor
from deploy.go2_onboard.model_loader import infer
from deploy.go2_onboard.navigation_observation import NavigationObservation

from .state import RuntimeOutput
from .supervisor import TerrainSupervisor


class PolicyRuntimeCore:
    """Simulator-independent SEA-Nav -> HIMLoco inference and safety path."""

    def __init__(self, navigation_policy, himloco_policy, device="cpu", supervisor=None):
        self.device = torch.device(device)
        self.navigation_policy = navigation_policy
        self.himloco_policy = himloco_policy
        self.nav_observation = NavigationObservation(self.device)
        self.him_observation = HIMLocoObservation(self.device)
        self.supervisor = supervisor or TerrainSupervisor()
        self.policy_to_motor = make_policy_to_motor()
        self.last_time = None
        self.control_hz = 0.0
        self.default_angles = torch.tensor([[.1, .8, -1.5, -.1, .8, -1.5,
                                             .1, 1.0, -1.5, -.1, 1.0, -1.5]], device=self.device)

    def reset(self):
        self.nav_observation.reset()
        self.him_observation.reset()
        self.supervisor.last_command[:] = 0

    def step(self, state):
        state.validate()
        now = time.perf_counter()
        if self.last_time is not None:
            self.control_hz = 1.0 / max(now - self.last_time, 1e-6)
        self.last_time = now
        gravity = torch.from_numpy(state.gravity).reshape(1, 3).to(self.device)
        angular = torch.from_numpy(state.angular_velocity).reshape(1, 3).to(self.device)
        linear = torch.from_numpy(state.linear_velocity).reshape(1, 3).to(self.device)
        q = torch.from_numpy(state.joint_position).reshape(1, 12).to(self.device)
        dq = torch.from_numpy(state.joint_velocity).reshape(1, 12).to(self.device)
        rays = torch.from_numpy(state.rays).reshape(1, 41).to(self.device)
        goal = torch.from_numpy(state.goal_xy).reshape(1, 2).to(self.device)
        zero_command = torch.zeros((1, 3), device=self.device) if self.supervisor.last_command is None else torch.from_numpy(self.supervisor.last_command).reshape(1, 3).to(self.device)
        nav_input = self.nav_observation.build(gravity, zero_command, linear, angular, rays, goal)
        t0 = time.perf_counter()
        raw = infer(self.navigation_policy, nav_input, 3)[0].detach().cpu().numpy()
        nav_ms = (time.perf_counter() - t0) * 1000
        supervised, terrain_state = self.supervisor.update(state, raw)
        command = torch.from_numpy(supervised).reshape(1, 3).to(self.device)
        him_input = self.him_observation.build(command, angular, gravity, q - self.default_angles, dq)
        t1 = time.perf_counter()
        him_action = infer(self.himloco_policy, him_input, 12)
        him_ms = (time.perf_counter() - t1) * 1000
        # The previous-action contract stores clipped policy actions, while the
        # actuator order is mapped separately at the final command boundary.
        self.him_observation.record_action(him_action)
        return RuntimeOutput(raw, supervised, him_action[0].detach().cpu().numpy(),
                             terrain_state.value, nav_ms, him_ms, self.control_hz,
                             float(np.linalg.norm(state.goal_xy)) < .45,
                             {"nav_observation_shape": tuple(nav_input.shape),
                              "himloco_observation_shape": tuple(him_input.shape),
                              "motor_action": him_action[0].detach().cpu().numpy()[self.policy_to_motor]})

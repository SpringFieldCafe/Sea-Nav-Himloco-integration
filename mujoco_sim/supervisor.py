from enum import Enum

import numpy as np


class TerrainState(str, Enum):
    NAVIGATE = "NAVIGATE"
    APPROACH_STAIRS = "APPROACH_STAIRS"
    ALIGN_STAIRS = "ALIGN_STAIRS"
    STAIRS_UP = "STAIRS_UP"
    STAIRS_DOWN = "STAIRS_DOWN"
    ROUGH_TERRAIN = "ROUGH_TERRAIN"
    RECOVERY = "RECOVERY"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class TerrainSupervisor:
    """Rules now use terrain truth; the input can later be a perception result."""

    HARD_LIMITS = np.asarray([1.0, 1.0, 2.0], dtype=np.float32)

    def __init__(self, speed_scale=1.0, command_filter_alpha=0.5,
                 open_space_assist=True):
        if speed_scale <= 0:
            raise ValueError("speed_scale must be positive")
        if not 0.0 < command_filter_alpha <= 1.0:
            raise ValueError("command_filter_alpha must be in (0, 1]")
        self.speed_scale = float(speed_scale)
        self.command_filter_alpha = float(command_filter_alpha)
        self.open_space_assist = bool(open_space_assist)
        self.state = TerrainState.NAVIGATE
        self.last_command = np.zeros(3, dtype=np.float32)

    def update(self, state, raw_command):
        terrain = state.terrain_hint
        if state.fallen or abs(state.roll) > 1.15 or abs(state.pitch) > 1.15:
            self.state = TerrainState.EMERGENCY_STOP
        elif terrain == "stairs_up":
            self.state = TerrainState.STAIRS_UP
        elif terrain == "stairs_down":
            self.state = TerrainState.STAIRS_DOWN
        elif terrain == "rough":
            self.state = TerrainState.ROUGH_TERRAIN
        elif terrain == "approach_stairs":
            self.state = TerrainState.APPROACH_STAIRS
        else:
            self.state = TerrainState.NAVIGATE

        base_limits = {
            TerrainState.NAVIGATE: (1.0, 1.0, 2.0),
            TerrainState.APPROACH_STAIRS: (.45, .30, .60),
            TerrainState.ALIGN_STAIRS: (.25, .18, .35),
            TerrainState.STAIRS_UP: (.45, .15, .50),
            TerrainState.STAIRS_DOWN: (.40, .12, .45),
            TerrainState.ROUGH_TERRAIN: (.50, .25, .55),
            TerrainState.RECOVERY: (0.0, 0.0, 0.0),
            TerrainState.EMERGENCY_STOP: (0.0, 0.0, 0.0),
        }[self.state]
        limits = np.minimum(np.asarray(base_limits, dtype=np.float32) * self.speed_scale,
                            self.HARD_LIMITS)
        scaled_command = np.asarray(raw_command, dtype=np.float32) * self.speed_scale
        command = np.clip(scaled_command, -limits, limits)
        if self.state == TerrainState.STAIRS_UP:
            # Stair risers can look like close obstacles to a flat-ground policy.
            # Keep a small forward bias so the robot climbs instead of reversing.
            command[0] = max(command[0], 0.12)
        elif self.state == TerrainState.STAIRS_DOWN:
            command[0] = max(command[0], 0.10)
        command = self._stabilize_open_space(state, command)
        command = (self.command_filter_alpha * command
                   + (1.0 - self.command_filter_alpha) * self.last_command)
        self.last_command = command
        return command, self.state

    def _stabilize_open_space(self, state, command):
        """Dampen incidental lateral drift only when the route is clear.

        The policy output remains unchanged and is logged as raw_command. This
        guard does not run near obstacles or when the goal is substantially
        off the body-forward axis, so it does not replace obstacle decisions.
        """
        if not self.open_space_assist or state.min_obstacle_distance < 2.5:
            return command
        goal_x, goal_y = np.asarray(state.goal_xy, dtype=np.float32)
        if goal_x <= 0.0:
            return command
        bearing = abs(float(np.arctan2(goal_y, max(goal_x, 1e-3))))
        center_weight = np.clip(1.0 - bearing / 0.35, 0.0, 1.0)
        if center_weight <= 0.0:
            return command
        adjusted = command.copy()
        # Preserve command signs and most of the model decision; only suppress
        # the unexplained lateral/yaw component near a centered clear goal.
        adjusted[1] *= 1.0 - 0.65 * center_weight
        adjusted[2] *= 1.0 - 0.35 * center_weight
        return adjusted

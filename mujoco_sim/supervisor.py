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

    def __init__(self, speed_scale=1.0, command_filter_alpha=0.5):
        if speed_scale <= 0:
            raise ValueError("speed_scale must be positive")
        if not 0.0 < command_filter_alpha <= 1.0:
            raise ValueError("command_filter_alpha must be in (0, 1]")
        self.speed_scale = float(speed_scale)
        self.command_filter_alpha = float(command_filter_alpha)
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
        command = np.clip(np.asarray(raw_command, dtype=np.float32), -limits, limits)
        command = (self.command_filter_alpha * command
                   + (1.0 - self.command_filter_alpha) * self.last_command)
        self.last_command = command
        return command, self.state

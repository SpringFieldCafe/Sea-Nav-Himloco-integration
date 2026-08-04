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

    def __init__(self):
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

        limits = {
            TerrainState.NAVIGATE: (1.0, 1.0, 2.0),
            TerrainState.APPROACH_STAIRS: (.35, .25, .5),
            TerrainState.ALIGN_STAIRS: (.15, .12, .25),
            TerrainState.STAIRS_UP: (.28, .12, .35),
            TerrainState.STAIRS_DOWN: (.24, .10, .30),
            TerrainState.ROUGH_TERRAIN: (.35, .20, .45),
            TerrainState.RECOVERY: (0.0, 0.0, 0.0),
            TerrainState.EMERGENCY_STOP: (0.0, 0.0, 0.0),
        }[self.state]
        command = np.clip(np.asarray(raw_command, dtype=np.float32), -np.asarray(limits), limits)
        command = .25 * command + .75 * self.last_command
        self.last_command = command
        return command, self.state

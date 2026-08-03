from typing import Dict, List


POLICY_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Unitree Go2 LowState motor order used by the HIMLoco deployment contract.
MOTOR_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def make_policy_to_motor() -> List[int]:
    motor_index = {name: i for i, name in enumerate(MOTOR_JOINT_NAMES)}
    mapping = [motor_index[name] for name in POLICY_JOINT_NAMES]
    if sorted(mapping) != list(range(12)):
        raise AssertionError(f"invalid policy_to_motor mapping: {mapping}")
    return mapping


def make_motor_to_policy() -> List[int]:
    policy_to_motor = make_policy_to_motor()
    inverse = [0] * len(policy_to_motor)
    for policy_index, motor_index in enumerate(policy_to_motor):
        inverse[motor_index] = policy_index
    if any(policy_to_motor[inverse[i]] != i for i in range(12)):
        raise AssertionError("joint mapping round-trip failed")
    return inverse

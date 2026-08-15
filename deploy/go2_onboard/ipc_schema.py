"""Versioned, one-way sensor packets for the split shadow runtime."""

import json
import math
from typing import Any, Dict

import numpy as np


SCHEMA_VERSION = 1
EXPECTED_SHAPES = {
    "joint_pos": (12,),
    "joint_vel": (12,),
    "imu_ang_vel": (3,),
    "projected_gravity": (3,),
    "base_linear_velocity_body": (3,),
    "base_angular_velocity_body": (3,),
    "lidar_rays": (41,),
    "goal_body": (2,),
}


def make_packet(
    sequence: int,
    timestamp_monotonic: float,
    joint_pos,
    joint_vel,
    imu_ang_vel,
    projected_gravity,
    base_linear_velocity_body,
    base_angular_velocity_body,
    lidar_rays,
    goal_body,
    sensor_age: Dict[str, Any],
    validity: Dict[str, bool],
    timestamp_wall: float,
) -> Dict[str, Any]:
    packet = {
        "schema_version": SCHEMA_VERSION,
        "sequence": int(sequence),
        "timestamp_monotonic": float(timestamp_monotonic),
        "timestamp_wall": float(timestamp_wall),
        "joint_pos": _array(joint_pos, "joint_pos"),
        "joint_vel": _array(joint_vel, "joint_vel"),
        "imu_ang_vel": _array(imu_ang_vel, "imu_ang_vel"),
        "projected_gravity": _array(projected_gravity, "projected_gravity"),
        "base_linear_velocity_body": _array(base_linear_velocity_body, "base_linear_velocity_body"),
        "base_angular_velocity_body": _array(base_angular_velocity_body, "base_angular_velocity_body"),
        "lidar_rays": _array(lidar_rays, "lidar_rays"),
        "goal_body": _array(goal_body, "goal_body"),
        "sensor_age": {str(k): _optional_float(v) for k, v in sensor_age.items()},
        "validity": {str(k): bool(v) for k, v in validity.items()},
    }
    validate_packet(packet)
    return packet


def encode_packet(packet: Dict[str, Any]) -> bytes:
    validate_packet(packet)
    return (json.dumps(packet, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def decode_packet(line: bytes) -> Dict[str, Any]:
    if not line:
        raise ValueError("empty IPC packet")
    packet = json.loads(line.decode("utf-8"))
    validate_packet(packet)
    return packet


def validate_packet(packet: Dict[str, Any]) -> None:
    if not isinstance(packet, dict):
        raise ValueError("IPC packet must be an object")
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported IPC schema version")
    if not isinstance(packet.get("sequence"), int) or packet["sequence"] < 0:
        raise ValueError("sequence must be a non-negative integer")
    if not math.isfinite(float(packet.get("timestamp_monotonic", float("nan")))):
        raise ValueError("timestamp_monotonic must be finite")
    for name, shape in EXPECTED_SHAPES.items():
        value = np.asarray(packet.get(name), dtype=np.float32)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    ages = packet.get("sensor_age")
    if not isinstance(ages, dict):
        raise ValueError("sensor_age must be an object")
    for value in ages.values():
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("sensor_age contains NaN or Inf")
    validity = packet.get("validity")
    if not isinstance(validity, dict):
        raise ValueError("validity must be an object")


def _array(value, name):
    array = np.asarray(value, dtype=np.float32)
    expected = EXPECTED_SHAPES[name]
    if array.shape != expected or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {expected}")
    return array.tolist()


def _optional_float(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("sensor age must be finite or null")
    return value

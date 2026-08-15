"""Torch-only Process B for one-way offline or real sensor shadowing."""

import argparse
import json
import socket
import time
from pathlib import Path

import torch

from .command_bridge import ReadOnlyCommandBridge
from .himloco_observation import HIMLocoObservation
from .ipc_schema import decode_packet
from .joint_mapping import make_policy_to_motor
from .model_loader import infer, load_himloco_policy, load_navigation_policy
from .navigation_observation import NavigationObservation


DEFAULT_SOCKET = "/tmp/sea_nav_shadow.sock"
DEFAULT_SEA = "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt"
DEFAULT_SEA_META = "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json"
DEFAULT_HIM = "models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt"


def run(args):
    device = torch.device(args.device)
    sea = load_navigation_policy(args.navigation_policy, args.navigation_metadata, device)
    him = load_himloco_policy(args.himloco_policy, device)
    nav_obs = NavigationObservation(device)
    him_obs = HIMLocoObservation(device)
    bridge = ReadOnlyCommandBridge([-1.0, -1.0, -2.0], [1.0, 1.0, 2.0], args.command_filter_alpha)
    policy_to_motor = make_policy_to_motor()
    last_command = torch.zeros((1, 3), dtype=torch.float32, device=device)
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log = open(args.log, "a", encoding="utf-8") if args.log else None
    started = time.monotonic()
    print(f"[shadow_worker] connecting socket={args.socket} device={device}")
    print(f"[shadow_worker] SEA sha256={sea.sha256}")
    print(f"[shadow_worker] HIM sha256={him.sha256}")
    print("[safety] one-way sensor IPC; no rclpy, LowCmd, SportClient, or command return path")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(args.connect_timeout)
    sock.connect(args.socket)
    sock.settimeout(1.0)
    buffer = b""
    last_sequence = None
    last_packet_received = None
    try:
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            chunk = sock.recv(65536)
            if not chunk:
                print("[shadow_worker] sensor bridge disconnected; stopping")
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    continue
                packet = decode_packet(line)
                if last_sequence is not None and packet["sequence"] <= last_sequence:
                    raise RuntimeError("IPC sequence did not increase")
                last_sequence = packet["sequence"]
                receive_time = time.perf_counter()
                packet_rate_hz = (
                    0.0 if last_packet_received is None
                    else 1.0 / max(receive_time - last_packet_received, 1e-6)
                )
                last_packet_received = receive_time
                loop_start = time.perf_counter()
                record, last_command = process_packet(
                    packet, sea, him, nav_obs, him_obs, bridge, policy_to_motor, device, last_command
                )
                loop_latency_ms = (time.perf_counter() - loop_start) * 1000.0
                record.update({
                    "mode": "shadow_worker",
                    "timestamp": time.time(),
                    "sequence": packet["sequence"],
                    "packet_rate_hz": packet_rate_hz,
                    "sensor_age": packet["sensor_age"],
                    "packet_validity": packet["validity"],
                    "goal_body": packet["goal_body"],
                    "loop_latency_ms": loop_latency_ms,
                    "deadline_miss": loop_latency_ms > 20.0,
                    "lowcmd_sent": False,
                })
                if log:
                    log.write(json.dumps(record, separators=(",", ":")) + "\n")
                    log.flush()
                print(json.dumps(record, separators=(",", ":")))
    except KeyboardInterrupt:
        print("[shadow_worker] stopped")
    finally:
        bridge.assert_no_writes()
        sock.close()
        if log:
            log.close()


def process_packet(packet, sea, him, nav_obs, him_obs, bridge, policy_to_motor, device, last_command):
    required = packet["validity"]
    if not all(required.get(name, False) for name in ("lowstate", "lidar", "odom", "goal")):
        return ({"runtime_state": "STALE_SENSOR", "fault_reason": "invalid_or_missing_sensor", "lowcmd_sent": False}, last_command)
    required_ages = ("lowstate", "lidar", "odom", "goal")
    if any(
        packet["sensor_age"].get(name) is None
        or packet["sensor_age"].get(name) > 0.25
        for name in required_ages
    ):
        return ({"runtime_state": "STALE_SENSOR", "fault_reason": "stale_sensor", "lowcmd_sent": False}, last_command)
    torch_packet = {key: torch.as_tensor(packet[key], dtype=torch.float32, device=device).reshape(1, -1)
                    for key in ("joint_pos", "joint_vel", "imu_ang_vel", "projected_gravity",
                                "base_linear_velocity_body", "base_angular_velocity_body", "lidar_rays", "goal_body")}
    q_policy = torch_packet["joint_pos"][:, policy_to_motor]
    dq_policy = torch_packet["joint_vel"][:, policy_to_motor]
    nav_input = nav_obs.build(
        torch_packet["projected_gravity"], last_command,
        torch_packet["base_linear_velocity_body"], torch_packet["base_angular_velocity_body"],
        torch_packet["lidar_rays"], torch_packet["goal_body"],
    )
    nav_start = time.perf_counter()
    nav_raw = infer(sea, nav_input, 3)
    nav_latency_ms = (time.perf_counter() - nav_start) * 1000.0
    raw_command = nav_raw[0].detach().cpu().numpy()
    filtered_command = bridge.filter(raw_command)
    command_tensor = torch.from_numpy(filtered_command).reshape(1, 3).to(device)
    him_input = him_obs.build(
        command_tensor, torch_packet["base_angular_velocity_body"], torch_packet["projected_gravity"],
        q_policy - torch.tensor([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], device=device).reshape(1, 12),
        dq_policy,
    )
    him_start = time.perf_counter()
    him_action = infer(him, him_input, 12)
    him_latency_ms = (time.perf_counter() - him_start) * 1000.0
    him_obs.record_action(him_action)
    return {
        "runtime_state": "SHADOW", "fault_reason": "", "sea_observation_shape": list(nav_input.shape),
        "sea_observation_finite": bool(torch.isfinite(nav_input).all()),
        "sea_observation_min": float(nav_input.min().item()),
        "sea_observation_max": float(nav_input.max().item()),
        "sea_raw_command": raw_command.tolist(), "sea_filtered_command": filtered_command.tolist(),
        "command_filter_alpha": bridge.filter_alpha, "sea_inference_latency_ms": nav_latency_ms,
        "him_observation_shape": list(him_input.shape), "him_observation_finite": bool(torch.isfinite(him_input).all()),
        "him_observation_min": float(him_input.min().item()),
        "him_observation_max": float(him_input.max().item()),
        "him_action": him_action[0].detach().cpu().tolist(),
        "him_action_min": float(him_action.min().item()),
        "him_action_max": float(him_action.max().item()),
        "him_inference_latency_ms": him_latency_ms,
        "lowcmd_sent": False,
    }, command_tensor


def build_parser():
    parser = argparse.ArgumentParser(description="Torch-only one-way SEA-Nav/HIMLoco shadow worker")
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "cuda:0"))
    parser.add_argument("--command-filter-alpha", type=float, default=0.15)
    parser.add_argument("--log", default="logs/go2_4d/shadow_worker.jsonl")
    parser.add_argument("--navigation-policy", default=DEFAULT_SEA)
    parser.add_argument("--navigation-metadata", default=DEFAULT_SEA_META)
    parser.add_argument("--himloco-policy", default=DEFAULT_HIM)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()

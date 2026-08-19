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
from .navigation_observation import NavigationObservation, clear_lidar_observation
from .timing import NumericStats, RateStats


DEFAULT_SOCKET = "/tmp/sea_nav_shadow.sock"
DEFAULT_SEA = "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt"
DEFAULT_SEA_META = "artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json"
DEFAULT_HIM = "models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt"


def dump_sea_observation(path, observation, packet):
    """Write one immutable copy of the exact observation passed to SEA-Nav."""
    before_dump = observation.detach().clone()
    payload = {
        "timestamp": time.time(),
        "sequence": int(packet["sequence"]),
        "shape": list(observation.shape),
        "goal_body": [float(value) for value in packet["goal_body"]],
        "odom_position": packet.get("odom_position"),
        "odom_yaw": packet.get("odom_yaw"),
        "lidar_rays": [float(value) for value in packet.get("lidar_rays", [])],
        "sea_observation": observation.detach().cpu().tolist(),
        "sea_observation_finite": bool(torch.isfinite(observation).all()),
        "history_order": "oldest_to_newest",
        "frame_dim": 55,
        "history_len": 10,
    }
    if not torch.equal(before_dump, observation.detach()):
        raise RuntimeError("SEA observation changed while preparing snapshot")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload


def snapshot_path(path, index, count):
    output = Path(path)
    if count <= 1:
        return output
    return output.parent / f"{output.stem}_{index:02d}{output.suffix or '.json'}"


def run(args):
    device = torch.device(args.device)
    sea = load_navigation_policy(args.navigation_policy, args.navigation_metadata, device)
    him = load_himloco_policy(args.himloco_policy, device)
    nav_obs = NavigationObservation(device)
    him_obs = HIMLocoObservation(device)
    bridge = ReadOnlyCommandBridge([-1.0, -1.0, -2.0], [1.0, 1.0, 2.0], args.command_filter_alpha)
    policy_to_motor = make_policy_to_motor()
    freshness = {
        "lowstate": args.lowstate_max_age,
        "odom": args.odom_max_age,
        "lidar": args.lidar_max_age,
    }
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
    rate_stats = RateStats()
    receive_wait_stats = NumericStats()
    decode_stats = NumericStats()
    sea_obs_stats = NumericStats()
    sea_inference_stats = NumericStats()
    him_obs_stats = NumericStats()
    him_inference_stats = NumericStats()
    loop_stats = NumericStats()
    log_stats = NumericStats()
    last_log_write_ms = 0.0
    last_summary = time.monotonic()
    deadline_miss_count = 0
    shadow_samples = 0
    sea_snapshot_count = 0
    last_sea_snapshot_at = None
    try:
        while args.duration <= 0 or time.monotonic() - started < args.duration:
            receive_start = time.perf_counter()
            chunk = sock.recv(65536)
            receive_wait_ms = (time.perf_counter() - receive_start) * 1000.0
            if not chunk:
                print("[shadow_worker] sensor bridge disconnected; stopping")
                break
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    continue
                packet_start = time.perf_counter()
                decode_start = time.perf_counter()
                packet = decode_packet(line)
                decode_ms = (time.perf_counter() - decode_start) * 1000.0
                decode_stats.add(decode_ms)
                if last_sequence is not None and packet["sequence"] <= last_sequence:
                    raise RuntimeError("IPC sequence did not increase")
                last_sequence = packet["sequence"]
                rate_stats.observe(packet_start)
                receive_wait_stats.add(receive_wait_ms)
                loop_start = time.perf_counter()
                record, last_command, sea_observation = process_packet(
                    packet, sea, him, nav_obs, him_obs, bridge, policy_to_motor, device,
                    last_command, freshness, args.assume_clear_lidar,
                )
                loop_latency_ms = (time.perf_counter() - loop_start) * 1000.0
                loop_total_ms = (time.perf_counter() - packet_start) * 1000.0
                deadline_miss = loop_total_ms > 20.0
                if deadline_miss:
                    deadline_miss_count += 1
                if record.get("runtime_state") == "SHADOW":
                    shadow_samples += 1
                    sea_obs_stats.add(record["sea_obs_build_ms"])
                    sea_inference_stats.add(record["sea_inference_latency_ms"])
                    him_obs_stats.add(record["him_obs_build_ms"])
                    him_inference_stats.add(record["him_inference_latency_ms"])
                    now_monotonic = time.monotonic()
                    interval_ready = (
                        last_sea_snapshot_at is None
                        or now_monotonic - last_sea_snapshot_at >= args.dump_sea_observation_interval_s
                    )
                    if (
                        args.dump_sea_observation
                        and sea_snapshot_count < args.dump_sea_observation_count
                        and shadow_samples >= args.dump_sea_observation_after_samples
                        and interval_ready
                    ):
                        sea_snapshot_count += 1
                        dump_sea_observation(
                            snapshot_path(
                                args.dump_sea_observation,
                                sea_snapshot_count,
                                args.dump_sea_observation_count,
                            ),
                            sea_observation,
                            packet,
                        )
                        last_sea_snapshot_at = now_monotonic
                loop_stats.add(loop_latency_ms)
                record.update({
                    "mode": "shadow_worker",
                    "timestamp": time.time(),
                    "sequence": packet["sequence"],
                    "packet_rate_hz": rate_stats.current_rate_hz(),
                    "ipc_receive_wait_ms": receive_wait_ms,
                    "decode_ms": decode_ms,
                    "sensor_age": packet["sensor_age"],
                    "packet_validity": packet["validity"],
                    "goal_body": packet["goal_body"],
                    "loop_latency_ms": loop_latency_ms,
                    "loop_total_ms": loop_total_ms,
                    "log_write_ms": last_log_write_ms,
                    "deadline_miss": deadline_miss,
                    "lowcmd_sent": False,
                })
                log_start = time.perf_counter()
                if log:
                    log.write(json.dumps(record, separators=(",", ":")) + "\n")
                last_log_write_ms = (time.perf_counter() - log_start) * 1000.0
                log_stats.add(last_log_write_ms)
                if time.monotonic() - last_summary >= args.summary_interval:
                    summary = {"mode": "shadow_worker", "state": record["runtime_state"], **rate_stats.summary()}
                    summary.update({
                        "receive_wait_p95_ms": receive_wait_stats.percentile(0.95),
                        "decode_p95_ms": decode_stats.percentile(0.95),
                        "sea_obs_p95_ms": sea_obs_stats.percentile(0.95),
                        "sea_p95_ms": sea_inference_stats.percentile(0.95),
                        "him_obs_p95_ms": him_obs_stats.percentile(0.95),
                        "him_p95_ms": him_inference_stats.percentile(0.95),
                        "loop_p95_ms": loop_stats.percentile(0.95),
                        "log_p95_ms": log_stats.percentile(0.95),
                        "deadline_misses": deadline_miss_count,
                        "lowcmd_sent": False,
                    })
                    print("[shadow_worker] " + json.dumps(summary, separators=(",", ":")))
                    last_summary = time.monotonic()
    except KeyboardInterrupt:
        print("[shadow_worker] stopped")
    finally:
        bridge.assert_no_writes()
        sock.close()
        if log:
            log.close()


def process_packet(packet, sea, him, nav_obs, him_obs, bridge, policy_to_motor, device,
                   last_command, freshness, assume_clear_lidar=False):
    required = packet["validity"]
    if not all(required.get(name, False) for name in ("lowstate", "lidar", "odom", "goal")):
        return ({"runtime_state": "STALE_SENSOR", "fault_reason": "invalid_or_missing_sensor", "lowcmd_sent": False}, last_command, None)
    # A fixed command-line Goal2D has no ROS receive age; validity still gates it.
    required_ages = ("lowstate", "lidar", "odom")
    if any(
        packet["sensor_age"].get(name) is None
        or packet["sensor_age"].get(name) > freshness[name]
        for name in required_ages
    ):
        return ({"runtime_state": "STALE_SENSOR", "fault_reason": "stale_sensor", "lowcmd_sent": False}, last_command, None)
    torch_packet = {key: torch.as_tensor(packet[key], dtype=torch.float32, device=device).reshape(1, -1)
                    for key in ("joint_pos", "joint_vel", "imu_ang_vel", "projected_gravity",
                                "base_linear_velocity_body", "base_angular_velocity_body", "lidar_rays", "goal_body")}
    if assume_clear_lidar:
        torch_packet["lidar_rays"] = torch.full_like(torch_packet["lidar_rays"], 5.0)
    q_policy = torch_packet["joint_pos"][:, policy_to_motor]
    dq_policy = torch_packet["joint_vel"][:, policy_to_motor]
    sea_obs_start = time.perf_counter()
    nav_input = nav_obs.build(
        torch_packet["projected_gravity"], last_command,
        torch_packet["base_linear_velocity_body"], torch_packet["base_angular_velocity_body"],
        torch_packet["lidar_rays"], torch_packet["goal_body"],
    )
    if assume_clear_lidar:
        nav_input = clear_lidar_observation(nav_input)
    sea_obs_build_ms = (time.perf_counter() - sea_obs_start) * 1000.0
    nav_start = time.perf_counter()
    nav_raw = infer(sea, nav_input, 3)
    nav_latency_ms = (time.perf_counter() - nav_start) * 1000.0
    raw_command = nav_raw[0].detach().cpu().numpy()
    filtered_command = bridge.filter(raw_command)
    command_tensor = torch.from_numpy(filtered_command).reshape(1, 3).to(device)
    him_obs_start = time.perf_counter()
    him_input = him_obs.build(
        command_tensor, torch_packet["base_angular_velocity_body"], torch_packet["projected_gravity"],
        q_policy - torch.tensor([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5], device=device).reshape(1, 12),
        dq_policy,
    )
    him_obs_build_ms = (time.perf_counter() - him_obs_start) * 1000.0
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
        "command_filter_alpha": bridge.filter_alpha, "sea_obs_build_ms": sea_obs_build_ms,
        "sea_inference_latency_ms": nav_latency_ms,
        "him_observation_shape": list(him_input.shape), "him_observation_finite": bool(torch.isfinite(him_input).all()),
        "him_observation_min": float(him_input.min().item()),
        "him_observation_max": float(him_input.max().item()),
        "him_obs_build_ms": him_obs_build_ms,
        "him_action": him_action[0].detach().cpu().tolist(),
        "him_action_min": float(him_action.min().item()),
        "him_action_max": float(him_action.max().item()),
        "him_inference_latency_ms": him_latency_ms,
        "lowcmd_sent": False,
    }, command_tensor, nav_input


def build_parser():
    parser = argparse.ArgumentParser(description="Torch-only one-way SEA-Nav/HIMLoco shadow worker")
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "cuda:0"))
    parser.add_argument("--summary-interval", type=float, default=1.0)
    parser.add_argument("--command-filter-alpha", type=float, default=0.15)
    parser.add_argument("--lowstate-max-age", type=float, default=0.10)
    parser.add_argument("--odom-max-age", type=float, default=0.10)
    parser.add_argument("--lidar-max-age", type=float, default=0.20)
    parser.add_argument("--log", default="logs/go2_4d/shadow_worker.jsonl")
    parser.add_argument("--navigation-policy", default=DEFAULT_SEA)
    parser.add_argument("--navigation-metadata", default=DEFAULT_SEA_META)
    parser.add_argument("--himloco-policy", default=DEFAULT_HIM)
    parser.add_argument("--dump-sea-observation", default=None)
    parser.add_argument("--dump-sea-observation-after-samples", type=int, default=100)
    parser.add_argument("--dump-sea-observation-count", type=int, default=1)
    parser.add_argument("--dump-sea-observation-interval-s", type=float, default=0.0)
    parser.add_argument("--assume-clear-lidar", action="store_true",
                        help="NO OBSTACLE AVOIDANCE: use 5m LiDAR rays")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dump_sea_observation_count < 1:
        raise SystemExit("--dump-sea-observation-count must be positive")
    if args.dump_sea_observation_interval_s < 0.0:
        raise SystemExit("--dump-sea-observation-interval-s must be non-negative")
    run(args)


if __name__ == "__main__":
    main()

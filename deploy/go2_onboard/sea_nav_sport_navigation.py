"""SEA-Nav command adapter for the Unitree high-level Sport/MPC service.

SEA-Nav supplies bounded ``vx/vy/vyaw`` commands.  This module deliberately
does not import LowCmd or write joint targets; when explicitly enabled it
forwards commands through Unitree's SportClient, whose firmware owns gait and
low-level stabilization.  Without ``--enable-motion`` it is a shadow check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .sea_nav_himloco_navigation import (
    DEFAULT_NAVIGATION_METADATA,
    DEFAULT_NAVIGATION_POLICY,
    NavigationMailbox,
    NavigationProcess,
    validate_navigation_model,
)
from .model_loader import infer
from .navigation_observation import NavigationObservation


def build_parser():
    parser = argparse.ArgumentParser(description="SEA-Nav via Unitree Sport/MPC")
    parser.add_argument("net")
    parser.add_argument("--sensor-socket", default="/tmp/sea_nav_shadow.sock")
    parser.add_argument("--navigation-policy", default=DEFAULT_NAVIGATION_POLICY)
    parser.add_argument("--navigation-metadata", default=DEFAULT_NAVIGATION_METADATA)
    parser.add_argument("--navigation-log", required=True)
    parser.add_argument("--navigation-vx-max", type=float, default=float("inf"))
    parser.add_argument("--fixed-sport-vx", type=float, default=None,
                        help="diagnostic fixed SportClient vx; no software speed cap")
    parser.add_argument("--navigation-vy-max", type=float, default=0.0)
    parser.add_argument("--navigation-filter-alpha", type=float, default=0.15)
    parser.add_argument("--navigation-hz", type=float, default=10.0)
    parser.add_argument("--navigation-command-max-age", type=float, default=0.25)
    parser.add_argument("--navigation-connect-timeout", type=float, default=10.0)
    parser.add_argument("--navigation-summary-interval", type=float, default=1.0)
    parser.add_argument("--goal-x", type=float)
    parser.add_argument("--goal-y", type=float)
    parser.add_argument("--goal-tolerance", type=float, default=0.15)
    parser.add_argument("--goal-reached-confirmations", type=int, default=5)
    parser.add_argument("--assume-clear-lidar", action="store_true")
    parser.add_argument("--max-sensor-age", type=float, default=0.10)
    parser.add_argument("--enable-motion", action="store_true",
                        help="explicitly forward commands to SportClient.Move")
    parser.add_argument("--duration", type=float, default=0.0)
    return parser


def _load_sport_client(net):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    ChannelFactoryInitialize(0, net)
    client = SportClient()
    client.SetTimeout(5.0)
    client.Init()
    return client


def _prepare_official_motion(client):
    """Enter the official standing/walking state before sending navigation commands."""
    prepare_started = time.monotonic()
    joystick_result = client.SwitchJoystick(False)
    print(f"[official-motion] SwitchJoystick(false) result={joystick_result}")
    if joystick_result not in (None, 0):
        raise RuntimeError(f"SportClient.SwitchJoystick(false) failed: {joystick_result}")
    time.sleep(0.5)
    stand_result = client.StandUp()
    print(f"[official-motion] StandUp result={stand_result}")
    if stand_result not in (None, 0):
        raise RuntimeError(f"SportClient.StandUp failed: {stand_result}")
    time.sleep(2.0)
    balance_result = client.BalanceStand()
    print(f"[official-motion] BalanceStand result={balance_result}")
    if balance_result not in (None, 0):
        raise RuntimeError(f"SportClient.BalanceStand failed: {balance_result}")
    time.sleep(1.0)
    print(json.dumps({
        "stage": "motion_prepare_complete",
        "elapsed_ms": (time.monotonic() - prepare_started) * 1000.0,
    }, separators=(",", ":")))


def _official_motion_diagnostics(client):
    """Query SDK-exposed read-only service facts; never changes motion state."""
    result = {"stage": "motion_service_diagnostics"}
    for name in ("GetApiVersion", "GetServerApiVersion", "GetLeaseId", "AutoRecoveryGet"):
        try:
            value = getattr(client, name)()
            if isinstance(value, tuple):
                value = list(value)
            result[name] = value
        except Exception as exc:
            result[name] = f"ERROR:{type(exc).__name__}:{exc}"
    print("[official-motion-state] " + json.dumps(
        result, separators=(",", ":"), default=str,
    ))


def official_mpc_contract_probe(loaded):
    """Build the documented 550-D input without touching any robot transport.

    The training contract stores projected gravity as a unit vector.  Keeping
    this probe here prevents a Point-LIO ``m/s^2`` gravity diagnostic from
    accidentally being fed to the navigation actor.
    """
    device = torch.device("cpu")
    observation_builder = NavigationObservation(device)
    observation = observation_builder.build(
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.full((1, 41), 5.0, dtype=torch.float32),
        torch.tensor([[2.0, 0.0]], dtype=torch.float32),
    )
    action = infer(loaded, observation, 3)[0].detach().cpu().numpy()
    if not np.isfinite(action).all():
        raise ValueError("official MPC contract probe returned non-finite action")
    return {
        "observation_shape": list(observation.shape),
        "gravity_convention": "unit_projected_gravity_z=-1",
        "probe_goal_body": [2.0, 0.0],
        "probe_action": action.astype(float).tolist(),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.goal_x is None) != (args.goal_y is None):
        raise SystemExit("provide both --goal-x and --goal-y")
    if not 0.0 < args.navigation_hz <= 20.0:
        raise SystemExit("--navigation-hz must be in (0,20]")
    fixed_vx = args.fixed_sport_vx
    if fixed_vx is not None:
        if not np.isfinite(fixed_vx):
            raise SystemExit("--fixed-sport-vx must be finite")
        loaded = None
        probe = {"mode": "FIXED_SPORT_DIAGNOSTIC", "fixed_command": [fixed_vx, 0.0, 0.0]}
    else:
        loaded = validate_navigation_model(args.navigation_policy, args.navigation_metadata)
        probe = official_mpc_contract_probe(loaded)
    mailbox = NavigationMailbox(args.navigation_command_max_age)
    config = {
        "sensor_socket": args.sensor_socket,
        "navigation_policy": str(Path(args.navigation_policy).expanduser().resolve()),
        "navigation_metadata": str(Path(args.navigation_metadata).expanduser().resolve()),
        "navigation_hz": args.navigation_hz,
        "navigation_command_max_age": args.navigation_command_max_age,
        "command_filter_alpha": args.navigation_filter_alpha,
        "navigation_vx_max": args.navigation_vx_max,
        "navigation_vy_max": args.navigation_vy_max,
        "connect_timeout": args.navigation_connect_timeout,
        "summary_interval": args.navigation_summary_interval,
        "navigation_log": args.navigation_log,
        "goal_x": args.goal_x,
        "goal_y": args.goal_y,
        "goal_tolerance": args.goal_tolerance,
        "goal_reached_confirmations": args.goal_reached_confirmations,
        "lowstate_max_age": args.max_sensor_age,
        "odom_max_age": 0.10,
        "lidar_max_age": 0.20,
        "assume_clear_lidar": args.assume_clear_lidar,
    }
    navigation = None if fixed_vx is not None else NavigationProcess(config, mailbox)
    client = _load_sport_client(args.net) if args.enable_motion else None
    print(f"[official] backend=unitree_sport_mpc enable_motion={args.enable_motion}")
    print("[official-contract] " + json.dumps(probe, separators=(",", ":")))
    print("[official] no LowCmd/joint target path")
    if not args.enable_motion:
        print("[official] SHADOW_ONLY: pass --enable-motion to call SportClient.Move")
    else:
        _prepare_official_motion(client)
        _official_motion_diagnostics(client)
    if navigation is not None:
        navigation.start()
    started = time.monotonic()
    try:
        period = 1.0 / args.navigation_hz
        command_sequence = 0
        while args.duration <= 0.0 or time.monotonic() - started < args.duration:
            if fixed_vx is not None:
                values = [float(fixed_vx), 0.0, 0.0]
                reason, age = "fixed_sport_diagnostic", 0.0
            else:
                command, reason, age = mailbox.current()
                if reason:
                    command = command.__class__(0.0, 0.0, 0.0)
                values = command.as_array().tolist()
            command_sequence += 1
            move_started = time.monotonic()
            if client is not None:
                result = client.Move(float(values[0]), float(values[1]), float(values[2]))
            else:
                result = "SHADOW"
            move_elapsed_ms = (time.monotonic() - move_started) * 1000.0
            print("[official-command] " + json.dumps({
                "sequence": command_sequence, "command": values,
                "reason": reason, "age_s": age, "result": result,
                "move_elapsed_ms": move_elapsed_ms,
            }, separators=(",", ":")))
            time.sleep(period)
    except KeyboardInterrupt:
        print("[official] Ctrl+C")
    finally:
        if client is not None:
            try:
                print(f"[official] StopMove result={client.StopMove()}")
            except Exception as exc:
                print(f"[official] StopMove failed: {type(exc).__name__}: {exc}")
        if navigation is not None:
            navigation.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

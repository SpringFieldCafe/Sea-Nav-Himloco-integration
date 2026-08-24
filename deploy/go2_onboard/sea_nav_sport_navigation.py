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

from .sea_nav_himloco_navigation import (
    DEFAULT_NAVIGATION_METADATA,
    DEFAULT_NAVIGATION_POLICY,
    NavigationMailbox,
    NavigationProcess,
    validate_navigation_model,
)


def build_parser():
    parser = argparse.ArgumentParser(description="SEA-Nav via Unitree Sport/MPC")
    parser.add_argument("net")
    parser.add_argument("--sensor-socket", default="/tmp/sea_nav_shadow.sock")
    parser.add_argument("--navigation-policy", default=DEFAULT_NAVIGATION_POLICY)
    parser.add_argument("--navigation-metadata", default=DEFAULT_NAVIGATION_METADATA)
    parser.add_argument("--navigation-log", required=True)
    parser.add_argument("--navigation-vx-max", type=float, default=0.15)
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.goal_x is None) != (args.goal_y is None):
        raise SystemExit("provide both --goal-x and --goal-y")
    if not 0.0 < args.navigation_hz <= 20.0:
        raise SystemExit("--navigation-hz must be in (0,20]")
    validate_navigation_model(args.navigation_policy, args.navigation_metadata)
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
    navigation = NavigationProcess(config, mailbox)
    client = _load_sport_client(args.net) if args.enable_motion else None
    print(f"[official] backend=unitree_sport_mpc enable_motion={args.enable_motion}")
    print("[official] no LowCmd/joint target path")
    if not args.enable_motion:
        print("[official] SHADOW_ONLY: pass --enable-motion to call SportClient.Move")
    navigation.start()
    started = time.monotonic()
    try:
        period = 1.0 / args.navigation_hz
        while args.duration <= 0.0 or time.monotonic() - started < args.duration:
            command, reason, age = mailbox.current()
            if reason:
                command = command.__class__(0.0, 0.0, 0.0)
            values = command.as_array().tolist()
            if client is not None:
                result = client.Move(float(values[0]), float(values[1]), float(values[2]))
            else:
                result = "SHADOW"
            print("[official-command] " + json.dumps({
                "command": values, "reason": reason, "age_s": age,
                "result": result,
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
        navigation.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

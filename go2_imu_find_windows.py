#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def pstdev(vals):
    if not vals:
        return float("nan")
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def norm3(x, y, z):
    return math.sqrt(x*x + y*y + z*z)


def load_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = [
            "receive_monotonic_time",
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",
            "linear_acceleration_x",
            "linear_acceleration_y",
            "linear_acceleration_z",
        ]
        missing = [k for k in required if k not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit("Missing CSV columns: " + ", ".join(missing))

        for r in reader:
            try:
                rows.append({
                    "t": float(r["receive_monotonic_time"]),
                    "gx": float(r["angular_velocity_x"]),
                    "gy": float(r["angular_velocity_y"]),
                    "gz": float(r["angular_velocity_z"]),
                    "ax": float(r["linear_acceleration_x"]),
                    "ay": float(r["linear_acceleration_y"]),
                    "az": float(r["linear_acceleration_z"]),
                })
            except (TypeError, ValueError):
                continue

    if not rows:
        raise SystemExit("No valid samples found.")
    t0 = rows[0]["t"]
    for r in rows:
        r["t"] -= t0
    return rows


def stats_for_rows(rs):
    gx = [r["gx"] for r in rs]
    gy = [r["gy"] for r in rs]
    gz = [r["gz"] for r in rs]
    ax = [r["ax"] for r in rs]
    ay = [r["ay"] for r in rs]
    az = [r["az"] for r in rs]

    return {
        "n": len(rs),
        "gstd": norm3(pstdev(gx), pstdev(gy), pstdev(gz)),
        "astd": norm3(pstdev(ax), pstdev(ay), pstdev(az)),
        "gmx": mean(gx),
        "gmy": mean(gy),
        "gmz": mean(gz),
        "amean_norm": norm3(mean(ax), mean(ay), mean(az)),
    }


def rows_in_window(rows, start, end):
    return [r for r in rows if start <= r["t"] < end]


def main():
    ap = argparse.ArgumentParser(
        description="Inspect Go2 IMU calibration CSV without numpy/pandas."
    )
    ap.add_argument(
        "--input",
        default="/tmp/go2_imu_calibration.csv",
        help="record_imu CSV path",
    )
    ap.add_argument(
        "--bin-seconds",
        type=float,
        default=1.0,
        help="per-bin summary interval (default: 1.0 s)",
    )
    ap.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="minimum samples per bin",
    )
    args = ap.parse_args()

    rows = load_rows(args.input)
    duration = rows[-1]["t"]

    print(f"INPUT={args.input}")
    print(f"SAMPLES={len(rows)}")
    print(f"DURATION_S={duration:.6f}")
    print()
    print(
        "sec        samples  gyro_std_norm  accel_std_norm  "
        "gyro_mean_x  gyro_mean_y  gyro_mean_z  accel_mean_norm"
    )
    print("-" * 112)

    n_bins = int(math.ceil(duration / args.bin_seconds))
    summaries = []

    for i in range(n_bins):
        start = i * args.bin_seconds
        end = start + args.bin_seconds
        rs = rows_in_window(rows, start, end)
        if len(rs) < args.min_samples:
            continue
        s = stats_for_rows(rs)
        summaries.append((start, end, s))
        print(
            f"{start:05.1f}-{end:05.1f} "
            f"{s['n']:7d} "
            f"{s['gstd']:14.6f} "
            f"{s['astd']:15.6f} "
            f"{s['gmx']:+12.6f} "
            f"{s['gmy']:+12.6f} "
            f"{s['gmz']:+12.6f} "
            f"{s['amean_norm']:15.6f}"
        )

    # Rank individual 1-second bins to make visual inspection easier.
    if summaries:
        static_rank = sorted(
            summaries,
            key=lambda x: (x[2]["gstd"] + 0.25 * x[2]["astd"])
        )[:8]

        rotate_rank = sorted(
            summaries,
            key=lambda x: abs(x[2]["gmz"]),
            reverse=True
        )[:8]

        print()
        print("LOWEST_MOTION_1S_BINS:")
        for start, end, s in static_rank:
            print(
                f"  {start:.1f}-{end:.1f}s  "
                f"gyro_std={s['gstd']:.6f}  "
                f"accel_std={s['astd']:.6f}  "
                f"mean_gz={s['gmz']:+.6f}"
            )

        print()
        print("STRONGEST_Z_ROTATION_1S_BINS:")
        for start, end, s in rotate_rank:
            print(
                f"  {start:.1f}-{end:.1f}s  "
                f"mean_gz={s['gmz']:+.6f}  "
                f"gyro_std={s['gstd']:.6f}  "
                f"accel_std={s['astd']:.6f}"
            )

    print()
    print("NEXT:")
    print("  Send the full output back. We can choose the real static and rotation windows")
    print("  from the measured data instead of assuming the dog moved exactly on schedule.")


if __name__ == "__main__":
    main()

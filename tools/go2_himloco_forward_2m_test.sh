#!/usr/bin/env bash
set -euo pipefail

# Fixed real-robot experiment entry point:
# SEA-Navigation 550->3 + HIMLoco 1460, forward 2 m, vx cap 2 m/s.
# go2_nav_start.sh performs the MCF/LIO/odom/sensor-bridge checks and
# releases MCF before launching the HIMLoco navigation process.

REPO="/home/hyz/桌面/sea_nav"

command -v cpupower >/dev/null 2>&1 || {
  printf '[FAIL] cpupower is required\n' >&2
  exit 1
}

printf '[GO2-HIMLOCO] setting CPU governor to performance\n'
sudo cpupower frequency-set -g performance

exec bash "$REPO/tools/go2_nav_start.sh" \
  --forward 2.0 \
  --speed 2.0

#!/usr/bin/env bash
set -euo pipefail

# One-click real-robot experiment entry point.
# The reliable official Unitree Sport/MPC + official LiDAR/odom path is now the
# default.  The previous direct HIMLoco/Point-LIO path remains available only
# with --backend himloco for controlled fallback comparisons.

REPO="/home/hyz/桌面/sea_nav"
SPEED="2.0"
BACKEND="official"
HIMLOCO="models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt"
GOAL_TOLERANCE="0.30"

if [[ -t 1 ]]; then
  RED=$'\033[1;31m'
  YELLOW=$'\033[1;33m'
  RESET=$'\033[0m'
else
  RED=''
  YELLOW=''
  RESET=''
fi

usage() {
  cat <<'EOF'
Usage:
  bash tools/go2_himloco_forward_2m_test.sh [--speed MPS]
  bash tools/go2_himloco_forward_2m_test.sh --backend himloco [--speed MPS] [--himloco PATH]

Default backend: official Unitree Sport/MPC + official cloud_base/cloud_deskewed/robot_odom.
--speed MPS       forward-vx upper bound, default: 2.0
--backend NAME    official (default) or himloco
--himloco PATH    fallback HIMLoco policy path; used only with --backend himloco
EOF
}

while (($#)); do
  case "$1" in
    --backend)
      (($# >= 2)) || { printf '[FAIL] --backend requires official or himloco\n' >&2; exit 1; }
      BACKEND="$2"
      shift 2
      ;;
    --speed)
      (($# >= 2)) || { printf '[FAIL] --speed requires a value\n' >&2; exit 1; }
      SPEED="$2"
      shift 2
      ;;
    --himloco)
      (($# >= 2)) || { printf '[FAIL] --himloco requires a path\n' >&2; exit 1; }
      HIMLOCO="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf '[FAIL] unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$BACKEND" in
  official|himloco) ;;
  *) printf '[FAIL] --backend must be official or himloco\n' >&2; exit 1 ;;
esac

command -v cpupower >/dev/null 2>&1 || {
  printf '[FAIL] cpupower is required\n' >&2
  exit 1
}

printf '%s[GO2] setting CPU governor to performance%s\n' "$YELLOW" "$RESET"
sudo cpupower frequency-set -g performance

if [[ "$BACKEND" == official ]]; then
  printf '%s[GO2] backend=official Sport/MPC; official LiDAR/odom; no Point-LIO%s\n' "$RED" "$RESET"
  printf '%s[GO2] safety: confirm the robot is standing in a clear area before motion starts%s\n' "$RED" "$RESET"
  exec bash "$REPO/tools/go2_seanav_navigation_test.sh" \
    --front-goal-distance 2.0 \
    --navigation-vx-max "$SPEED" \
    --goal-tolerance "$GOAL_TOLERANCE" \
    --enable-official-motion \
    --non-interactive
fi

printf '%s[GO2] backend=himloco fallback; this path uses Point-LIO and direct LowCmd%s\n' "$YELLOW" "$RESET"
exec bash "$REPO/tools/go2_nav_start.sh" \
  --forward 2.0 \
  --speed "$SPEED" \
  --himloco "$HIMLOCO"

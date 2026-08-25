#!/usr/bin/env bash
set -euo pipefail

# One-click real-robot experiment entry point:
# SEA-Navigation 550->3 stays fixed; HIMLoco policy and navigation vx cap are
# explicit command-line parameters. The goal remains forward 2 m.
# go2_nav_start.sh performs the MCF/LIO/odom/sensor-bridge checks and
# releases MCF before launching the HIMLoco navigation process.

REPO="/home/hyz/桌面/sea_nav"
SPEED="2.0"
HIMLOCO="models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt"

if [[ -t 1 ]]; then
  YELLOW=$'\033[1;33m'
  RESET=$'\033[0m'
else
  YELLOW=''
  RESET=''
fi

usage() {
  cat <<'EOF'
Usage:
  bash tools/go2_himloco_forward_2m_test.sh [--speed MPS] [--himloco PATH]

The SEA-Nav navigation model is intentionally fixed to the repository default.
--speed MPS       forward-vx upper bound, default: 2.0
--himloco PATH    HIMLoco policy path, default: models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt
EOF
}

while (($#)); do
  case "$1" in
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

command -v cpupower >/dev/null 2>&1 || {
  printf '[FAIL] cpupower is required\n' >&2
  exit 1
}

printf '%s[GO2-HIMLOCO] setting CPU governor to performance%s\n' "$YELLOW" "$RESET"
sudo cpupower frequency-set -g performance

exec bash "$REPO/tools/go2_nav_start.sh" \
  --forward 2.0 \
  --speed "$SPEED" \
  --himloco "$HIMLOCO"

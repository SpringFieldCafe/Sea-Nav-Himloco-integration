#!/usr/bin/env bash
set -euo pipefail

cd /home/hyz/桌面/sea_nav
HIMLOCO_PYTHON=/home/hyz/anaconda3/envs/himloco/bin/python

"$HIMLOCO_PYTHON" -c 'import numpy, torch, unitree_sdk2py; import deploy.go2_onboard.himloco_fixed_control' || {
    echo "[preflight] HIMLOCO_PYTHON import check failed" >&2
    exit 1
}
echo "[preflight] HIMLOCO_PYTHON=$HIMLOCO_PYTHON imports=PASS"

exec "$HIMLOCO_PYTHON" -m deploy.go2_onboard.himloco_fixed_control \
    enp3s0 \
    --policy models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt \
    --vx 0.0 \
    --vy 0.0 \
    --wz 0.0 \
    --max-sensor-age 0.10 \
    --hold-duration 5.0

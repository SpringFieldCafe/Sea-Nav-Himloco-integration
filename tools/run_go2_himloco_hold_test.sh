#!/usr/bin/env bash
set -euo pipefail

cd /home/hyz/桌面/sea_nav
source /home/hyz/anaconda3/etc/profile.d/conda.sh
conda activate himloco

exec python -m deploy.go2_onboard.himloco_fixed_control \
    enp3s0 \
    --policy models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt \
    --vx 0.0 \
    --vy 0.0 \
    --wz 0.0 \
    --max-sensor-age 0.10 \
    --hold-duration 5.0

#!/usr/bin/env bash
set -euo pipefail

MODE=""
BAG="/tmp/lio_bags/good_candidate/transformed_input"
LOG=""
DOMAIN="78"
RATE="1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARAMS="$ROOT/lio/sea_nav_lio_ws/install_humble_clean/point_lio_unilidar/share/point_lio_unilidar/config/sea_nav_go2.yaml"

usage() {
    printf 'usage: %s --mode A|B [--bag PATH] [--log PATH] [--domain ID] [--rate R]\n' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --bag) BAG="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --domain) DOMAIN="$2"; shift 2 ;;
        --rate) RATE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

[[ "$MODE" == A || "$MODE" == B ]] || { usage >&2; exit 2; }
[[ -d "$BAG" ]] || { printf 'missing bag: %s\n' "$BAG" >&2; exit 1; }
[[ -f "$PARAMS" ]] || { printf 'missing Point-LIO params: %s\n' "$PARAMS" >&2; exit 1; }

LOG="${LOG:-/tmp/lio_deskew_mode_${MODE}.log}"
ROS_LOG_DIR="/tmp/ros_log_deskew_mode_${MODE}"
mkdir -p "$ROS_LOG_DIR"

set +u
source /opt/ros/humble/setup.bash
source /home/hyz/unitree_msgs_humble_ws/install/setup.bash
source "$ROOT/lio/sea_nav_lio_ws/install_humble_clean/setup.bash"
set -u

export ROS_DOMAIN_ID="$DOMAIN"
export ROS_LOCALHOST_ONLY=1
export ROS_LOG_DIR
unset RMW_IMPLEMENTATION

PIDS=()
cleanup() {
    trap - INT TERM EXIT
    for pid in "${PIDS[@]:-}"; do
        kill -INT "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${PIDS[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

POINTLIO_TOPIC="/sea_nav/lio/transformed_cloud"
POINTLIO_ARGS=(ros2 run point_lio_unilidar pointlio_mapping
    --ros-args --params-file "$PARAMS")
if [[ "$MODE" == B ]]; then
    /usr/bin/python3 "$ROOT/tools/ros2_lidar_deskew/deskew_node.py" >>"$LOG" 2>&1 &
    PIDS+=("$!")
    POINTLIO_TOPIC="/sea_nav/lio/deskewed_cloud"
    POINTLIO_ARGS+=(--remap "/sea_nav/lio/transformed_cloud:=/sea_nav/lio/deskewed_cloud")
fi

"${POINTLIO_ARGS[@]}" >>"$LOG" 2>&1 &
PIDS+=("$!")
sleep 3

printf 'MODE=%s\nPOINTLIO_INPUT_TOPIC=%s\nBAG=%s\nLOG=%s\n' \
    "$MODE" "$POINTLIO_TOPIC" "$BAG" "$LOG"
ros2 bag play "$BAG" --topics \
    /sea_nav/lio/transformed_cloud /sea_nav/lio/transformed_raw_imu --rate "$RATE"
sleep 2
printf 'RESULT_MODE_%s\n' "$MODE"
rg 'STATE_POSITION_DRIFT_FROM_INIT_M|STATE_VELOCITY_NORM|DESKEW_DIAG' "$LOG" | tail -12 || true

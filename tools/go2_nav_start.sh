#!/usr/bin/env bash
set -Eeuo pipefail

# Single user entry for the guarded SEA-Nav + HIMLoco deployment.
# The LIO selfcheck owns the MCF release gate and starts only read-only
# preparation components.  This script never presses Start/A or sends motion.

REPO="/home/hyz/桌面/sea_nav"
SELF_CHECK="$REPO/tools/go2_lio_selfcheck.sh"
HIMLOCO_PYTHON="/home/hyz/anaconda3/envs/himloco/bin/python"
HIMLOCO_DEFAULT="$REPO/models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt"
NAV_POLICY_DEFAULT="$REPO/artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt"
NAV_METADATA_DEFAULT="$REPO/artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json"
SOCKET="/tmp/sea_nav_shadow.sock"
NAV_MAX_AGE="0.25"
NAV_FILTER_ALPHA="0.15"
NAV_HZ="10"
SENSOR_MAX_AGE="0.10"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ID="$STAMP"
RUN_ROOT="$REPO/logs/go2_nav_start/$RUN_ID"
SELF_LOG="$RUN_ROOT/lio_selfcheck.log"
NAV_LOG="$REPO/logs/go2_navigation/navigation.jsonl"
EVENT_TRACE="/tmp/sea_nav_navigation_event_trace.jsonl"
SELF_MARKER="$RUN_ROOT/selfcheck.started"
NAV_PID_FILE="$RUN_ROOT/navigation.pid"
OWNERSHIP_FILE="$RUN_ROOT/ownership.txt"

LEGACY_POLICY_1_SHA="456218effd3a4befdbcd54f85c4474c1aa282df1dd81894bd7761543c18dd11a"
HIMLOCO_1460_SHA="cab2489dda7732a7d6f51595aa6362384445c738537d0c1c91569054c7b9f5d1"

GOAL_X=""
GOAL_Y=""
FORWARD=""
SPEED="inf"
HIMLOCO="$HIMLOCO_DEFAULT"
NAV_POLICY="$NAV_POLICY_DEFAULT"
SELF_PIPE_PID=""
SELF_ROOT=""
CLEANUP_STARTED=0
CLEANUP_FAILED=0

die() { printf '[NAV-START] ERROR: %s\n' "$*" >&2; exit 1; }
say() { printf '[NAV-START] %s\n' "$*"; }

usage() {
  cat <<'EOF'
Usage:
  tools/go2_nav_start.sh --forward METERS [--speed MPS] [--himloco PATH] [--nav-policy PATH]
  tools/go2_nav_start.sh --goal-x X --goal-y Y [--speed MPS] [--himloco PATH] [--nav-policy PATH]

--speed is the navigation vx upper bound; use inf to disable the speed limit.
Start/A remain manual. B remains the formal controller ESTOP.
EOF
}

while (($#)); do
  case "$1" in
    --himloco)
      (($# >= 2)) || die "--himloco requires a path"
      HIMLOCO="$2"; shift 2 ;;
    --goal-x)
      (($# >= 2)) || die "--goal-x requires a value"
      GOAL_X="$2"; shift 2 ;;
    --goal-y)
      (($# >= 2)) || die "--goal-y requires a value"
      GOAL_Y="$2"; shift 2 ;;
    --forward)
      (($# >= 2)) || die "--forward requires meters"
      FORWARD="$2"; shift 2 ;;
    --speed)
      (($# >= 2)) || die "--speed requires m/s"
      SPEED="$2"; shift 2 ;;
    --nav-policy)
      (($# >= 2)) || die "--nav-policy requires a path"
      NAV_POLICY="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -x "$HIMLOCO_PYTHON" ]] || die "missing HIMLoco Python: $HIMLOCO_PYTHON"
[[ -x "$SELF_CHECK" ]] || die "missing selfcheck: $SELF_CHECK"
command -v gnome-terminal >/dev/null 2>&1 || die "gnome-terminal is required by the existing deployment entry"
command -v ros2 >/dev/null 2>&1 || die "missing ros2"
command -v timeout >/dev/null 2>&1 || die "missing timeout"

set +u
source /opt/ros/humble/setup.bash
source /home/hyz/unitree_msgs_humble_ws/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export CYCLONEDDS_URI='<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="enp3s0" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'

is_number() {
  awk -v value="$1" 'BEGIN {
    if (value !~ /^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/) exit 1
    n = value + 0
    exit !(n == n && n < 1e300 && n > -1e300)
  }'
}

if [[ "$SPEED" != inf ]]; then
  is_number "$SPEED" || die "--speed must be finite or inf"
fi
if [[ "$SPEED" != inf ]]; then
  awk -v speed="$SPEED" 'BEGIN { exit !(speed >= 0.0) }' \
    || die "--speed must be non-negative or inf"
fi

if [[ -n "$FORWARD" ]]; then
  [[ -z "$GOAL_X" && -z "$GOAL_Y" ]] || die "use --forward or --goal-x/--goal-y, not both"
  is_number "$FORWARD" || die "--forward must be finite"
  awk -v forward="$FORWARD" 'BEGIN { exit !(forward > 0.0) }' \
    || die "--forward must be greater than zero"
else
  [[ -n "$GOAL_X" && -n "$GOAL_Y" ]] || die "provide --forward or both --goal-x and --goal-y"
  is_number "$GOAL_X" || die "--goal-x must be finite"
  is_number "$GOAL_Y" || die "--goal-y must be finite"
fi
REQUESTED_GOAL_X="$GOAL_X"
REQUESTED_GOAL_Y="$GOAL_Y"

existing_selfcheck_supervisors() {
  ps -eo pid=,args= | awk -v self="$$" '
    $1 != self && $0 ~ /go2_lio_selfcheck\.sh/ {print}
  '
}

resolve_repo_path() {
  if [[ "$1" = /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$REPO" "$1"; fi
}

HIMLOCO="$(resolve_repo_path "$HIMLOCO")"
NAV_POLICY="$(resolve_repo_path "$NAV_POLICY")"
[[ -f "$HIMLOCO" ]] || die "missing HIMLoco policy: $HIMLOCO"
[[ -f "$NAV_POLICY" ]] || die "missing SEA-Nav policy: $NAV_POLICY"

if [[ "$NAV_POLICY" == "$NAV_POLICY_DEFAULT" ]]; then
  NAV_METADATA="$NAV_METADATA_DEFAULT"
else
  NAV_METADATA="${NAV_POLICY%.*}.json"
fi
[[ -f "$NAV_METADATA" ]] || die "missing navigation metadata: $NAV_METADATA"

STALE_SELF_CHECKS="$(existing_selfcheck_supervisors || true)"
if [[ -n "$STALE_SELF_CHECKS" ]]; then
  say "STALE_SUPERVISOR: an existing selfcheck supervisor owns another run"
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    stale_pid="${line%% *}"
    stale_cmd="${line#* }"
    stale_root="$(printf '%s\n' "$stale_cmd" | grep -o '/logs/go2_lio_selfcheck/[^ ]*' | head -n 1 || true)"
    say "STALE_SUPERVISOR_PID=$stale_pid RUN_ROOT=${stale_root:-UNKNOWN} CMD=$stale_cmd"
  done <<<"$STALE_SELF_CHECKS"
  die "STALE_SUPERVISOR (stop the listed selfcheck manually and rerun)"
fi

HIM_SHA="$(sha256sum "$HIMLOCO" | awk '{print $1}')"
case "$HIM_SHA" in
  "$HIMLOCO_1460_SHA")
    POLICY_OVERRIDE=(); POLICY_PROFILE="himloco_1460" ;;
  "$LEGACY_POLICY_1_SHA")
    POLICY_OVERRIDE=(--allow-policy-override); POLICY_PROFILE="legacy_policy_1" ;;
  *)
    die "HIMLoco SHA is not an approved 1460 or policy_1 model: $HIM_SHA" ;;
esac

SELF_CHECK_ARGS=()
if [[ -n "$FORWARD" ]]; then
  SELF_CHECK_ARGS=(--forward "$FORWARD")
else
  SELF_CHECK_ARGS=(--goal-x "$GOAL_X" --goal-y "$GOAL_Y")
fi

if pgrep -af 'deploy\.go2_onboard\.sea_nav_himloco_navigation' >/dev/null 2>&1; then
  die "formal SEA-Nav process already exists; stop it manually before rerun"
fi

mkdir -p "$RUN_ROOT"
touch "$SELF_MARKER"
say "policy_profile=$POLICY_PROFILE sha256=$HIM_SHA"
if [[ -n "$FORWARD" ]]; then
  say "GOAL_MODE=FORWARD"
  say "FORWARD=$FORWARD"
  say "WORLD_GOAL=PENDING_ODOM"
else
  say "GOAL_MODE=WORLD"
  say "goal_world=[$GOAL_X,$GOAL_Y]"
fi
say "navigation_vx_max=$SPEED"
say "starting existing MCF/LIO/adapter/sensor_bridge selfcheck"

# Keep the existing central supervisor as the preparation owner. It performs
# the explicit CheckMode -> conditional ReleaseMode -> CheckMode gate and
# starts no control process. Avoid a pipeline so Ctrl+C reaches this PID.
(cd "$REPO" && exec "$SELF_CHECK" --managed "${SELF_CHECK_ARGS[@]}") >"$SELF_LOG" 2>&1 &
SELF_PIPE_PID=$!
printf 'RUN_ID=%s\nSELF_CHECK_PID=%s\n' "$RUN_ID" "$SELF_PIPE_PID" >"$OWNERSHIP_FILE"
say "SELFCHECK_PID=$SELF_PIPE_PID log=$SELF_LOG"

latest_selfcheck_root() {
  find "$REPO/logs/go2_lio_selfcheck" -mindepth 2 -maxdepth 2 \
    -type f -name status.txt -newer "$SELF_MARKER" -printf '%T@ %h\n' 2>/dev/null \
    | sort -nr | awk 'NR == 1 { print $2 }'
}

status_value() {
  local file="$1" key="$2"
  [[ -f "$file" ]] || return 1
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$file"
}

goal_status_error() {
  local message="$1" file="$SELF_ROOT/goal.status"
  say "GOAL_STATUS_FILE=$file"
  say "GOAL_STATUS_CONTENT_BEGIN"
  if [[ -f "$file" ]]; then
    sed 's/^/[NAV-START] /' "$file" >&2
  else
    say "<missing>"
  fi
  say "GOAL_STATUS_CONTENT_END"
  die "$message"
}

wait_pid_exit() {
  local pid="$1" loops="${2:-50}"
  for _ in $(seq 1 "$loops"); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  return 1
}

stop_navigation() {
  local pid
  say "[STOP] NAVIGATION"
  if [[ ! -f "$NAV_PID_FILE" ]]; then
    say "[PASS] NAVIGATION_NOT_STARTED"
    return 0
  fi
  pid="$(cat "$NAV_PID_FILE" 2>/dev/null || true)"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    say "[PASS] NAVIGATION_ALREADY_STOPPED pid=${pid:-unknown}"
    return 0
  fi
  say "STOPPING_FORMAL_CONTROLLER pid=$pid"
  kill -INT "$pid" 2>/dev/null || true
  if wait_pid_exit "$pid" 80; then
    say "[PASS] NAVIGATION_STOPPED"
    return 0
  fi
  say "[WARN] NAVIGATION pid=$pid did not exit after SIGINT; sending SIGTERM"
  kill -TERM "$pid" 2>/dev/null || true
  if wait_pid_exit "$pid" 50; then
    say "[PASS] NAVIGATION_STOPPED_AFTER_SIGTERM"
    return 0
  fi
  say "[WARN] NAVIGATION pid=$pid still alive; escalating to SIGKILL"
  kill -KILL "$pid" 2>/dev/null || true
  if wait_pid_exit "$pid" 20; then
    say "[PASS] NAVIGATION_STOPPED_AFTER_SIGKILL"
    return 0
  fi
  say "[FAIL] NAVIGATION_STOP_FAILED pid=$pid log=$NAV_LOG"
  return 1
}

stop_owned_component() {
  local label="$1" status_file="$2" pid_file="$3" log_file="$4"
  local state pid
  say "[STOP] $label"
  state="$(status_value "$status_file" STATE 2>/dev/null || true)"
  if [[ "$state" == REUSED ]]; then
    say "[PASS] ${label}_NOT_OWNED (REUSED)"
    return 0
  fi
  [[ "$state" == RUNNING ]] || {
    say "[PASS] ${label}_NOT_RUNNING"
    return 0
  }
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    say "[PASS] ${label}_ALREADY_STOPPED pid=${pid:-unknown}"
    return 0
  fi
  kill -INT "$pid" 2>/dev/null || true
  if wait_pid_exit "$pid" 50; then
    say "[PASS] ${label}_STOPPED"
    return 0
  fi
  say "[WARN] $label pid=$pid did not exit after SIGINT; log=$log_file"
  kill -TERM "$pid" 2>/dev/null || true
  if wait_pid_exit "$pid" 30; then
    say "[PASS] ${label}_STOPPED_AFTER_SIGTERM"
    return 0
  fi
  say "[WARN] $label pid=$pid still alive; escalating to SIGKILL"
  kill -KILL "$pid" 2>/dev/null || true
  if wait_pid_exit "$pid" 20; then
    say "[PASS] ${label}_STOPPED_AFTER_SIGKILL"
    return 0
  fi
  say "[FAIL] ${label}_STOP_FAILED pid=$pid log=$log_file"
  return 1
}

stop_owned_pid() {
  local label="$1" pid="$2" log_file="$3"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  say "[STOP] $label pid=$pid"
  kill -INT "$pid" 2>/dev/null || true
  wait_pid_exit "$pid" 50 && { say "[PASS] ${label}_STOPPED"; return 0; }
  say "[WARN] $label pid=$pid did not exit after SIGINT; log=$log_file"
  kill -TERM "$pid" 2>/dev/null || true
  wait_pid_exit "$pid" 30 && { say "[PASS] ${label}_STOPPED_AFTER_SIGTERM"; return 0; }
  say "[WARN] $label pid=$pid still alive; escalating to SIGKILL"
  kill -KILL "$pid" 2>/dev/null || true
  wait_pid_exit "$pid" 20 && { say "[PASS] ${label}_STOPPED_AFTER_SIGKILL"; return 0; }
  say "[FAIL] ${label}_STOP_FAILED pid=$pid log=$log_file"
  return 1
}

cleanup() {
  local status_dir point_state point_launch point_child
  (( CLEANUP_STARTED == 0 )) || return 0
  CLEANUP_STARTED=1
  trap - INT TERM EXIT
  echo
  say "[STOP] CLEANUP RUN_ID=$RUN_ID"
  stop_navigation || CLEANUP_FAILED=1
  status_dir="$SELF_ROOT"
  if [[ -n "$status_dir" ]]; then
    stop_owned_component "DIAGNOSTIC" "$status_dir/diagnostic.status" "$status_dir/diagnostic.pid" "$status_dir/diagnostic.log" || CLEANUP_FAILED=1
    stop_owned_component "SENSOR_BRIDGE" "$status_dir/sensor_bridge.status" "$status_dir/sensor_bridge.pid" "$status_dir/sensor_bridge.log" || CLEANUP_FAILED=1
    stop_owned_component "ADAPTER" "$status_dir/adapter.status" "$status_dir/adapter.pid" "$status_dir/adapter.log" || CLEANUP_FAILED=1
    stop_owned_component "POINT_LIO" "$status_dir/point_lio.status" "$status_dir/point_lio.pid" "$status_dir/point_lio.log" || CLEANUP_FAILED=1
    point_state="$(status_value "$status_dir/point_lio.status" STATE 2>/dev/null || true)"
    point_launch="$(status_value "$status_dir/point_lio.status" LAUNCH_PID 2>/dev/null || true)"
    point_child="$(status_value "$status_dir/point_lio.status" CHILD_PID 2>/dev/null || true)"
    if [[ "$point_state" == RUNNING && "$point_child" =~ ^[0-9]+$ && "$point_child" != "$point_launch" ]]; then
      stop_owned_pid "POINT_LIO_CHILD" "$point_child" "$status_dir/point_lio.log" || CLEANUP_FAILED=1
    fi
  fi
  if [[ "$SELF_PIPE_PID" =~ ^[0-9]+$ ]] && kill -0 "$SELF_PIPE_PID" 2>/dev/null; then
    say "[STOP] SELFCHECK"
    kill -INT "$SELF_PIPE_PID" 2>/dev/null || true
    if wait_pid_exit "$SELF_PIPE_PID" 50; then
      say "[PASS] SELFCHECK_STOPPED"
    else
      say "[WARN] SELFCHECK pid=$SELF_PIPE_PID did not exit after SIGINT"
      kill -TERM "$SELF_PIPE_PID" 2>/dev/null || true
      if wait_pid_exit "$SELF_PIPE_PID" 30; then
        say "[PASS] SELFCHECK_STOPPED_AFTER_SIGTERM"
      else
        say "[FAIL] SELFCHECK_STOP_FAILED pid=$SELF_PIPE_PID log=$SELF_LOG"
        CLEANUP_FAILED=1
      fi
    fi
  fi
  if [[ -f "$SELF_LOG" ]] && grep -q 'CLEAN_SHUTDOWN=FAIL' "$SELF_LOG"; then
    say "SELF_CHECK_CLEANUP=FAIL"
    CLEANUP_FAILED=1
  elif [[ -f "$SELF_LOG" ]] && grep -q 'CLEAN_SHUTDOWN=PASS' "$SELF_LOG"; then
    say "SELF_CHECK_CLEANUP=PASS"
  fi
  if (( CLEANUP_FAILED == 0 )); then
    say "CLEAN_SHUTDOWN=PASS"
  else
    say "CLEAN_SHUTDOWN=FAIL"
  fi
}

on_signal() {
  cleanup
  exit 130
}

trap on_signal INT TERM
trap cleanup EXIT

selfcheck_ready() {
  local root="$1" status mcf
  status="$root/status.txt"
  mcf="$root/mcf.status"
  [[ -f "$status" && -f "$mcf" ]] || return 1
  grep -Eq '^RAW_LOWSTATE[[:space:]]*= PASS$' "$status" || return 1
  grep -Eq '^RAW_CLOUD[[:space:]]*= PASS$' "$status" || return 1
  grep -Eq '^RAW_IMU[[:space:]]*= PASS$' "$status" || return 1
  grep -Eq '^POINT_LIO_DATA[[:space:]]*= PASS$' "$status" || return 1
  grep -Eq '^POINT_LIO_PUBS[[:space:]]*= 1$' "$status" || return 1
  grep -Eq '^ODOM_BASE_DATA[[:space:]]*= PASS$' "$status" || return 1
  grep -Eq '^ODOM_BASE_PUBS[[:space:]]*= 1$' "$status" || return 1
  grep -Eq '^ODOM_BASE_FRAME[[:space:]]*= odom -> base_link$' "$status" || return 1
  grep -Eq '^SENSOR_BRIDGE[[:space:]]*= RUNNING$' "$status" || return 1
  grep -Eq '^BRIDGE_ODOM_SUB[[:space:]]*= PASS$' "$status" || return 1
  grep -q "^MCF_FINAL_MODE=''$" "$mcf" || return 1
  [[ -f "$root/goal.status" ]] || return 1
  [[ -n "$(status_value "$root/goal.status" GOAL_X || true)" ]] || return 1
  [[ -n "$(status_value "$root/goal.status" GOAL_Y || true)" ]] || return 1
}

selfcheck_running() {
  local state
  [[ "$SELF_PIPE_PID" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$SELF_PIPE_PID" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$SELF_PIPE_PID" 2>/dev/null || true)"
  [[ -n "$state" && "$state" != Z* ]]
}

selfcheck_early_exit() {
  say "SELFCHECK_EARLY_EXIT=YES"
  if [[ -f "$SELF_LOG" ]]; then
    say "SELFCHECK_LOG_TAIL_BEGIN=$SELF_LOG"
    tail -n 60 "$SELF_LOG"
    say "SELFCHECK_LOG_TAIL_END"
  else
    say "SELFCHECK_LOG_MISSING=$SELF_LOG"
  fi
  exit 1
}

SELF_ROOT=""
for _ in $(seq 1 180); do
  selfcheck_running || selfcheck_early_exit
  SELF_ROOT="$(latest_selfcheck_root || true)"
  if [[ -n "$SELF_ROOT" ]] && selfcheck_ready "$SELF_ROOT"; then
    say "MCF/LIO/adapter/sensor_bridge preparation PASS: $SELF_ROOT"
    break
  fi
  sleep 1
done

if [[ -z "$SELF_ROOT" ]] || ! selfcheck_ready "$SELF_ROOT"; then
  say "preparation did not reach PASS; formal navigation was not started"
  [[ -f "$SELF_LOG" ]] && tail -n 40 "$SELF_LOG" || true
  exit 1
fi

printf 'SELF_CHECK_PID=%s\nDIAGNOSTIC_PID=%s\nSENSOR_BRIDGE_PID=%s\nADAPTER_PID=%s\nPOINT_LIO_PID=%s\n' \
  "$SELF_PIPE_PID" \
  "$(cat "$SELF_ROOT/diagnostic.pid" 2>/dev/null || true)" \
  "$(cat "$SELF_ROOT/sensor_bridge.pid" 2>/dev/null || true)" \
  "$(cat "$SELF_ROOT/adapter.pid" 2>/dev/null || true)" \
  "$(cat "$SELF_ROOT/point_lio.pid" 2>/dev/null || true)" >>"$OWNERSHIP_FILE"
printf 'POINT_LIO_LAUNCH_PID=%s\nPOINT_LIO_CHILD_PID=%s\n' \
  "$(status_value "$SELF_ROOT/point_lio.status" LAUNCH_PID || true)" \
  "$(status_value "$SELF_ROOT/point_lio.status" CHILD_PID || true)" >>"$OWNERSHIP_FILE"

read_latest_odom() {
  local sample values qx qy qz qw
  sample="$(timeout 5s ros2 topic echo /sea_nav/lio/odom_base --once 2>/dev/null)" \
    || die "unable to read fresh /sea_nav/lio/odom_base"
  values="$(printf '%s\n' "$sample" | awk '
    /position:/ { block="position"; next }
    /orientation:/ { block="orientation"; next }
    block == "position" && $1 == "x:" { px=$2 }
    block == "position" && $1 == "y:" { py=$2 }
    block == "orientation" && $1 == "x:" { qx=$2 }
    block == "orientation" && $1 == "y:" { qy=$2 }
    block == "orientation" && $1 == "z:" { qz=$2 }
    block == "orientation" && $1 == "w:" { qw=$2 }
    END {
      if (px == "" || py == "" || qx == "" || qy == "" || qz == "" || qw == "") exit 1
      print px, py, qx, qy, qz, qw
    }')" || die "unable to parse /sea_nav/lio/odom_base"
  read -r CURRENT_X CURRENT_Y qx qy qz qw <<<"$values"
  CURRENT_YAW_DEG="$(/usr/bin/python3 - "$qx" "$qy" "$qz" "$qw" <<'PY'
import math
import sys

qx, qy, qz, qw = map(float, sys.argv[1:])
yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
print(f"{math.degrees(yaw):.9f}")
PY
  )" || die "unable to calculate odometry yaw"
}

read_latest_odom
GOAL_STATUS_FILE="$SELF_ROOT/goal.status"
[[ -f "$GOAL_STATUS_FILE" ]] || goal_status_error "missing current-run goal.status"
SENSOR_BRIDGE_GOAL_X="$(status_value "$GOAL_STATUS_FILE" GOAL_X || true)"
SENSOR_BRIDGE_GOAL_Y="$(status_value "$GOAL_STATUS_FILE" GOAL_Y || true)"

numbers_close() {
  /usr/bin/python3 - "$1" "$2" <<'PY'
import math
import sys

a, b = map(float, sys.argv[1:])
raise SystemExit(0 if math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-6 else 1)
PY
}

STATUS_GOAL_MODE="$(status_value "$GOAL_STATUS_FILE" GOAL_MODE || true)"
[[ -n "$STATUS_GOAL_MODE" ]] || goal_status_error "goal.status has no GOAL_MODE"
[[ "$STATUS_GOAL_MODE" == "FORWARD" && -n "$FORWARD" || \
   "$STATUS_GOAL_MODE" == "WORLD" && -z "$FORWARD" ]] \
  || goal_status_error "goal mode mismatch between launcher and selfcheck"
if [[ -n "$FORWARD" ]]; then
  STATUS_FORWARD="$(status_value "$GOAL_STATUS_FILE" FORWARD || true)"
  [[ -n "$STATUS_FORWARD" ]] || goal_status_error "goal.status has no FORWARD"
  numbers_close "$FORWARD" "$STATUS_FORWARD" || goal_status_error "forward value differs from goal.status"
else
  numbers_close "$REQUESTED_GOAL_X" "$SENSOR_BRIDGE_GOAL_X" \
    || goal_status_error "requested goal X differs from goal.status"
  numbers_close "$REQUESTED_GOAL_Y" "$SENSOR_BRIDGE_GOAL_Y" \
    || goal_status_error "requested goal Y differs from goal.status"
fi
[[ -n "$SENSOR_BRIDGE_GOAL_X" && -n "$SENSOR_BRIDGE_GOAL_Y" ]] \
  || goal_status_error "goal.status has no world goal"
GOAL_X="$SENSOR_BRIDGE_GOAL_X"
GOAL_Y="$SENSOR_BRIDGE_GOAL_Y"

GOAL_MODE="$STATUS_GOAL_MODE"
DRIFT_CHECK="$(grep -E '^DRIFT_CHECK[[:space:]]*=' "$SELF_ROOT/status.txt" | tail -n 1 | sed 's/.*=[[:space:]]*//')"
[[ -n "$DRIFT_CHECK" ]] || DRIFT_CHECK="UNKNOWN"
say "HIMLOCO_POLICY=$HIMLOCO"
say "NAV_POLICY=$NAV_POLICY"
say "CURRENT_X=$CURRENT_X"
say "CURRENT_Y=$CURRENT_Y"
say "CURRENT_YAW_DEG=$CURRENT_YAW_DEG"
say "GOAL_STATUS_FILE=$GOAL_STATUS_FILE"
say "FORWARD=${FORWARD:-<none>}"
say "GOAL_MODE=$GOAL_MODE"
say "GOAL_X=$GOAL_X"
say "GOAL_Y=$GOAL_Y"
say "SENSOR_BRIDGE_GOAL_X=$SENSOR_BRIDGE_GOAL_X"
say "SENSOR_BRIDGE_GOAL_Y=$SENSOR_BRIDGE_GOAL_Y"
say "NAVIGATION_GOAL_X=$GOAL_X"
say "NAVIGATION_GOAL_Y=$GOAL_Y"
say "GOAL_CONTRACT=PASS"
say "NAVIGATION_VX_MAX=$SPEED"
say "MCF=PASS"
say "LIO=PASS"
say "ADAPTER=PASS"
say "BRIDGE=PASS"
say "DRIFT_CHECK=$DRIFT_CHECK"
say "DRIFT_GATE=WARNING_ONLY"
say "READY_FOR_MANUAL_START=YES"

NAV_TITLE="Go2 SEA-Nav + HIMLoco (manual Start/A; B=ESTOP)"
NAV_COMMAND="cd $(printf '%q' "$REPO"); set +u; source /opt/ros/humble/setup.bash; source /home/hyz/unitree_msgs_humble_ws/install/setup.bash; set -u; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; export ROS_DOMAIN_ID=0; export ROS_LOCALHOST_ONLY=0; export CYCLONEDDS_URI='<CycloneDDS><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"enp3s0\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>'; printf '[NAV-START] preparation PASS; Start/A remain manual; B=ESTOP\\n'; set +e; $(printf '%q' "$HIMLOCO_PYTHON") -m deploy.go2_onboard.sea_nav_himloco_navigation enp3s0 --policy $(printf '%q' "$HIMLOCO") --vx 0 --vy 0 --wz 0 --goal-x $(printf '%q' "$GOAL_X") --goal-y $(printf '%q' "$GOAL_Y") --sensor-socket $(printf '%q' "$SOCKET") --navigation-policy $(printf '%q' "$NAV_POLICY") --navigation-metadata $(printf '%q' "$NAV_METADATA") --navigation-hz "$NAV_HZ" --navigation-command-max-age "$NAV_MAX_AGE" --navigation-filter-alpha "$NAV_FILTER_ALPHA" --navigation-vx-max "$SPEED" --hold-duration 0 --max-sensor-age "$SENSOR_MAX_AGE" --navigation-log $(printf '%q' "$NAV_LOG") --event-trace $(printf '%q' "$EVENT_TRACE") ${POLICY_OVERRIDE[*]-} & nav_pid=\$!; printf '%s\\n' \$nav_pid > $(printf '%q' "$NAV_PID_FILE"); wait \$nav_pid; rc=\$?; rm -f $(printf '%q' "$NAV_PID_FILE"); printf '[NAV-START] navigation exited rc=%s; no automatic restart\\n' "\$rc"; exit"

gnome-terminal --title="$NAV_TITLE" -- bash -lc "$NAV_COMMAND"
for _ in $(seq 1 20); do
  [[ -s "$NAV_PID_FILE" ]] && break
  sleep 0.1
done
if [[ -s "$NAV_PID_FILE" ]]; then
  printf 'FORMAL_NAVIGATION_PID=%s\n' "$(cat "$NAV_PID_FILE")" >>"$OWNERSHIP_FILE"
else
  say "[WARN] formal navigation PID file not observed yet: $NAV_PID_FILE"
fi
say "formal navigation terminal started; no Start/A was sent"
say "selfcheck central supervisor remains active; logs: $RUN_ROOT"

wait "$SELF_PIPE_PID" 2>/dev/null || true

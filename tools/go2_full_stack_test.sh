#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot Go2 sensor/LIO/HIMLoco test entry point.
# HIMLoco is opt-in; Start/A remain manual and this script never sends them.

ROOT="/home/hyz/桌面/sea_nav"
LIO_WS="$ROOT/lio/sea_nav_lio_ws"
ROS_SETUP="/opt/ros/humble/setup.bash"
UNITREE_SETUP="/home/hyz/unitree_msgs_humble_ws/install/setup.bash"
LIO_SETUP="$LIO_WS/install_humble_clean/setup.bash"
SDK_PYTHON="/home/hyz/anaconda3/envs/himloco/bin/python"
NET="enp3s0"
RUN_ROOT="/tmp/go2_full_stack_test/$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"

POLICY="$ROOT/models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt"
RUN_HIMLOCO=0
KEEP_ALIVE=0
CPU_RESULT=FAIL
MCF_RESULT=FAIL
SENSOR_RESULT=FAIL
TRANSFORM_RESULT=FAIL
LIO_RESULT=FAIL
DESKEW_RESULT=FAIL
HIMLOCO_RESULT=NOT_REQUESTED
CLEANUP_START=FAIL
CLEANUP_END=FAIL
CLEANED=0

mkdir -p "$LOG_ROOT" "$PID_ROOT"

say() { printf '[GO2] %s\n' "$*"; }
pass() { printf '[PASS] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; }

source_ros() {
  # Humble setup scripts read unset variables under nounset.
  set +u
  source "$ROS_SETUP"
  source "$UNITREE_SETUP"
  source "$LIO_SETUP"
  set -u
}

write_pid() {
  local name="$1" pid="$2"
  printf '%s\n' "$pid" >"$PID_ROOT/$name.pid"
}

pid_alive() { kill -0 "$1" 2>/dev/null; }

wait_dead() {
  local pid="$1" deadline=$((SECONDS + 5))
  while pid_alive "$pid" && (( SECONDS < deadline )); do sleep 0.2; done
  ! pid_alive "$pid"
}

stop_owned_pid() {
  local label="$1" pid_file="$2" signal="${3:-INT}" pid
  [[ -f "$pid_file" ]] || return 0
  pid="$(<"$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if ! pid_alive "$pid"; then
    return 0
  fi
  say "[STOP] $label pid=$pid signal=$signal"
  kill -"$signal" "$pid" 2>/dev/null || true
  if ! wait_dead "$pid"; then
    say "[STOP] $label still alive; escalating SIGTERM pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
    if ! wait_dead "$pid"; then
      say "[STOP] $label still alive; escalating SIGKILL pid=$pid"
      kill -KILL "$pid" 2>/dev/null || true
      wait_dead "$pid" || true
    fi
  fi
  if pid_alive "$pid"; then
    fail "${label}_STOPPED=FAIL pid=$pid"
  else
    pass "${label}_STOPPED"
  fi
}

cleanup_old_processes() {
  say "[0] CLEAN OLD PROCESSES"
  # These patterns are limited to this project's process names. The current
  # shell command line does not contain any of them and is never targeted.
  pkill -TERM -f 'deploy\.go2_onboard\.sensor_bridge' 2>/dev/null || true
  pkill -TERM -f 'transform_everything' 2>/dev/null || true
  pkill -TERM -f 'sea_nav_lidar_deskew' 2>/dev/null || true
  pkill -TERM -f 'sea_nav_point_lio' 2>/dev/null || true
  pkill -TERM -f 'pointlio_mapping' 2>/dev/null || true
  sleep 3

  local residual
  residual="$(pgrep -af 'sensor_bridge|transform_everything|deskew|point_lio|pointlio' || true)"
  if [[ -n "$residual" ]]; then
    printf '%s\n' "$residual"
    say "[CLEANUP] residual process exists"
    pkill -TERM -f 'deploy\.go2_onboard\.sensor_bridge' 2>/dev/null || true
    pkill -TERM -f 'transform_everything' 2>/dev/null || true
    pkill -TERM -f 'sea_nav_lidar_deskew' 2>/dev/null || true
    pkill -TERM -f 'sea_nav_point_lio' 2>/dev/null || true
    pkill -TERM -f 'pointlio_mapping' 2>/dev/null || true
    sleep 3
  fi

  residual="$(pgrep -af 'sensor_bridge|transform_everything|deskew|point_lio|pointlio' || true)"
  if [[ -n "$residual" ]]; then
    printf '%s\n' "$residual"
    say "[CLEANUP] residual process exists after second pass"
    CLEANUP_START=FAIL
  else
    CLEANUP_START=PASS
  fi
}

wait_ros_graph_cleanup() {
  local attempt nodes
  for attempt in {1..5}; do
    nodes="$(ros2 node list 2>/dev/null || true)"
    if ! grep -Eq '^/(sea_nav_lidar_deskew|sea_nav_point_lio|sea_nav_sensor_transform)$' <<<"$nodes"; then
      return 0
    fi
    sleep 1
  done
  say "[CLEANUP] warning: ROS graph still contains old project nodes"
  printf '%s\n' "$nodes"
  return 1
}

cleanup() {
  local rc=$?
  (( CLEANED )) && return "$rc"
  CLEANED=1
  say "[CLEANUP] stopping only this run's owned PIDs"
  stop_owned_pid HIMLOCO "$PID_ROOT/himloco.pid" INT || true
  stop_owned_pid POINT_LIO "$PID_ROOT/pointlio.pid" INT || true
  stop_owned_pid TRANSFORM "$PID_ROOT/transform.pid" INT || true
  stop_owned_pid POINT_LIO_LAUNCH "$PID_ROOT/launch.pid" INT || true
  stop_owned_pid SENSOR_BRIDGE "$PID_ROOT/sensor_bridge.pid" TERM || true
  if wait_ros_graph_cleanup; then
    CLEANUP_END=PASS
  else
    CLEANUP_END=FAIL
  fi
  printf '\n========== GO2 TEST SUMMARY ==========\n'
  printf 'CPU: %s\n' "$CPU_RESULT"
  printf 'MCF: %s\n' "$MCF_RESULT"
  printf 'Sensor: %s\n' "$SENSOR_RESULT"
  printf 'Transform: %s\n' "$TRANSFORM_RESULT"
  printf 'Point-LIO: %s\n' "$LIO_RESULT"
  printf 'Deskew: %s\n' "$DESKEW_RESULT"
  printf 'HIMLoco: %s\n' "$HIMLOCO_RESULT"
  printf 'CLEANUP_START: %s\n' "$CLEANUP_START"
  printf 'CLEANUP_END: %s\n' "$CLEANUP_END"
  printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
  return "$rc"
}
trap cleanup EXIT INT TERM

die() { fail "$*"; exit 1; }

check_cpu_governor() {
  say "[CPU] cpupower frequency-info"
  command -v cpupower >/dev/null 2>&1 || { fail "cpupower not found"; die "CPU governor check failed"; }
  cpupower frequency-info 2>&1 | tee "$LOG_ROOT/cpupower.log" || true
  local bad=0 file governor
  shopt -s nullglob
  local policies=(/sys/devices/system/cpu/cpufreq/policy*/scaling_governor)
  shopt -u nullglob
  if ((${#policies[@]} == 0)); then
    bad=1
  else
    for file in "${policies[@]}"; do
      governor="$(<"$file")"
      [[ "$governor" == performance ]] || bad=1
    done
  fi
  if (( bad )); then
    fail "CPU governor is not performance"
    printf 'sudo cpupower frequency-set -g performance\n' >&2
    die "CPU governor gate failed"
  fi
  CPU_RESULT=PASS
  pass "CPU governor performance"
}

check_mcf() {
  [[ -x "$SDK_PYTHON" ]] || die "MCF check Python missing: $SDK_PYTHON"
  say "[MCF] checking MotionSwitcher mode"
  local output
  if ! output="$($SDK_PYTHON - "$NET" <<'PY'
import sys
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

net = sys.argv[1]
def mode_name(data):
    return str((data or {}).get("name", "") or "")

try:
    ChannelFactoryInitialize(0, net)
    client = MotionSwitcherClient()
    client.SetTimeout(5.0)
    client.Init()
    code, data = client.CheckMode()
    if code != 0:
        print(f"MCF_CHECK_CODE={code}")
        raise SystemExit(2)
    initial = mode_name(data)
    print(f"INITIAL_MODE={initial}")
    if initial == "":
        print("RELEASE_REQUEST=NOT_NEEDED")
        print("RELEASE_RESULT=PASS")
        print("FINAL_MODE=")
        raise SystemExit(0)
    if initial != "mcf":
        print("RELEASE_REQUEST=NOT_REQUESTED")
        print("RELEASE_RESULT=FAIL")
        print(f"FINAL_MODE={initial}")
        raise SystemExit(3)
    print("RELEASE_REQUEST=REQUESTED")
    release_code, _ = client.ReleaseMode()
    if release_code != 0:
        print(f"RELEASE_CODE={release_code}")
        print("RELEASE_RESULT=FAIL")
        print("FINAL_MODE=mcf")
        raise SystemExit(4)
    final_code, final_data = client.CheckMode()
    final = mode_name(final_data)
    print(f"FINAL_CHECK_CODE={final_code}")
    print("RELEASE_RESULT=PASS" if final_code == 0 and final == "" else "RELEASE_RESULT=FAIL")
    print(f"FINAL_MODE={final}")
    raise SystemExit(0 if final_code == 0 and final == "" else 5)
except SystemExit:
    raise
except Exception as exc:
    print(f"MCF_EXCEPTION={type(exc).__name__}: {exc}")
    raise SystemExit(6)
PY
  )"; then
    printf '%s\n' "$output" | tee "$LOG_ROOT/mcf.log"
    die "MCF check/release failed"
  fi
  printf '%s\n' "$output" | tee "$LOG_ROOT/mcf.log"
  grep -q '^FINAL_MODE=$' "$LOG_ROOT/mcf.log" || die "MCF final mode is not empty"
  if grep -q '^RELEASE_REQUEST=NOT_NEEDED$' "$LOG_ROOT/mcf.log"; then
    say "MCF already released"
  fi
  MCF_RESULT=PASS
}

topic_rate() {
  local topic="$1" output rate
  output="$(timeout 6s ros2 topic hz "$topic" 2>&1 || true)"
  rate="$(printf '%s\n' "$output" | awk '/average rate:/ {print $3}' | tail -1)"
  [[ "$rate" =~ ^[0-9]+([.][0-9]+)?$ ]] && printf '%s\n' "$rate" || printf '0\n'
}

check_rate() {
  local topic="$1" minimum="$2" label="$3" rate
  rate="$(topic_rate "$topic")"
  printf '[Sensor] %s rate=%.3fHz required>=%.3fHz\n' "$label" "$rate" "$minimum"
  awk -v rate="$rate" -v minimum="$minimum" 'BEGIN { exit !(rate >= minimum) }'
}

start_sensor_bridge() {
  say "[START] sensor_bridge"
  /usr/bin/python3 -m deploy.go2_onboard.sensor_bridge \
    --duration 0 \
    --log "$LOG_ROOT/sensor_bridge.jsonl" \
    --lowstate-topic /lowstate \
    --lidar-topic /utlidar/cloud \
    --odom-topic /sea_nav/lio/odom_base \
    >"$LOG_ROOT/sensor_bridge.log" 2>&1 &
  write_pid sensor_bridge "$!"
  sleep 2
  pid_alive "$(<"$PID_ROOT/sensor_bridge.pid")" || die "sensor_bridge exited early"
  if check_rate /utlidar/cloud 10 cloud && check_rate /utlidar/imu 100 imu; then
    SENSOR_RESULT=PASS
    pass "raw sensor rates"
  else
    die "raw sensor rate check failed"
  fi
}

find_descendant_by_text() {
  local root="$1" needle="$2" child cmd
  for child in $(pgrep -P "$root" 2>/dev/null || true); do
    cmd="$(ps -o args= -p "$child" 2>/dev/null || true)"
    if [[ "$cmd" == *"$needle"* ]]; then
      printf '%s\n' "$child"
      return 0
    fi
    find_descendant_by_text "$child" "$needle" && return 0
  done
  return 1
}

start_lio() {
  say "[START] Point-LIO launch (includes transform_everything and deskew)"
  ros2 launch sea_nav_lio_bringup point_lio_go2.launch.py deskew:=true \
    >"$LOG_ROOT/point_lio.log" 2>&1 &
  local launch_pid="$!" transform_pid point_pid
  write_pid launch "$launch_pid"
  sleep 5
  pid_alive "$launch_pid" || die "Point-LIO launch exited early"
  transform_pid="$(find_descendant_by_text "$launch_pid" transform_everything || true)"
  point_pid="$(find_descendant_by_text "$launch_pid" pointlio_mapping || true)"
  [[ "$transform_pid" =~ ^[0-9]+$ ]] || die "transform_everything child not found"
  [[ "$point_pid" =~ ^[0-9]+$ ]] || die "pointlio_mapping child not found"
  write_pid transform "$transform_pid"
  write_pid pointlio "$point_pid"
  if ros2 node list 2>/dev/null | grep -q '^/sea_nav_point_lio$'; then
    LIO_RESULT=PASS
    pass "Point-LIO node"
  else
    die "Point-LIO node not found"
  fi
  if ros2 node list 2>/dev/null | grep -q '^/sea_nav_lidar_deskew$'; then
    DESKEW_RESULT=PASS
  else
    DESKEW_RESULT=FAIL
    die "deskew node not found"
  fi
  if check_rate /sea_nav/lio/transformed_raw_imu 100 transformed_imu; then
    TRANSFORM_RESULT=PASS
    pass "transformed IMU"
  else
    die "transformed IMU rate check failed"
  fi
  printf '%s\n' 'LIO STATE'
  grep '\[LIO-DIAG\] STATE ' "$LOG_ROOT/point_lio.log" | tail -1 || true
  printf '%s\n' 'DESKEW_STATUS'
  grep 'DESKEW_STATUS' "$LOG_ROOT/point_lio.log" | tail -3 || true
}

start_himloco() {
  (( RUN_HIMLOCO )) || return 0
  [[ -x "$SDK_PYTHON" ]] || die "HIMLOCO Python missing"
  [[ -f "$POLICY" ]] || die "policy not found: $POLICY"
  "$SDK_PYTHON" -c 'import numpy, torch, unitree_sdk2py; import deploy.go2_onboard.himloco_fixed_control' \
    >"$LOG_ROOT/himloco_preflight.log" 2>&1 || die "HIMLOCO import preflight failed"
  say "[START] HIMLoco policy=$POLICY"
  "$SDK_PYTHON" -m deploy.go2_onboard.himloco_fixed_control \
    "$NET" --policy "$POLICY" --vx 0.1 --vy 0 --wz 0 --max-sensor-age 0.10 \
    >"$LOG_ROOT/himloco.log" 2>&1 &
  write_pid himloco "$!"
  HIMLOCO_RESULT=STARTED_MANUAL_START_A
}

usage() {
  cat <<'EOF'
Usage: tools/go2_full_stack_test.sh [--himloco] [--policy POLICY.pt] [--keep-alive]

The default run stops after sensor/LIO/deskew checks. HIMLoco is opt-in;
Start/A remain manual and B remains the controller ESTOP.
With --keep-alive, the checked background processes remain running until Ctrl+C.
EOF
}

keep_alive_until_signal() {
  trap 'cleanup; exit 130' INT TERM
  trap - EXIT
  say "KEEP_ALIVE MODE"
  say "Background processes are running"
  printf 'sensor_bridge PID=%s\n' "$(<"$PID_ROOT/sensor_bridge.pid")"
  printf 'transform PID=%s\n' "$(<"$PID_ROOT/transform.pid")"
  printf 'pointlio PID=%s\n' "$(<"$PID_ROOT/pointlio.pid")"
  while :; do
    sleep 1
  done
}

main() {
  while (($#)); do
    case "$1" in
      --himloco) RUN_HIMLOCO=1; shift ;;
      --policy) (($# >= 2)) || die "--policy needs a value"; POLICY="$2"; shift 2 ;;
      --keep-alive) KEEP_ALIVE=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  cd "$ROOT"
  cleanup_old_processes
  source_ros
  wait_ros_graph_cleanup || say "[CLEANUP] warning: old ROS graph entries remain; continuing"
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export ROS_DOMAIN_ID=0
  export ROS_LOCALHOST_ONLY=0
  printf 'RMW=%s\nROS_DOMAIN_ID=%s\nROS_LOCALHOST_ONLY=%s\n' \
    "$RMW_IMPLEMENTATION" "$ROS_DOMAIN_ID" "$ROS_LOCALHOST_ONLY"
  check_cpu_governor
  check_mcf
  start_sensor_bridge
  start_lio
  start_himloco
  if (( KEEP_ALIVE )); then
    keep_alive_until_signal
  fi
  say "READY: Start/A remain manual; B is ESTOP"
}

main "$@"

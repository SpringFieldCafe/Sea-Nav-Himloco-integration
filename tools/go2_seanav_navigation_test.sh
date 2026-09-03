#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot SEA-Nav -> HIMLoco Go2 validation entry point.  The controller
# retains the existing manual Start/A and B ESTOP safety gates.

ROOT="/home/hyz/桌面/sea_nav"
LIO_WS="$ROOT/lio/sea_nav_lio_ws"
ROS_SETUP="/opt/ros/humble/setup.bash"
UNITREE_SETUP="/home/hyz/unitree_msgs_humble_ws/install/setup.bash"
LIO_BASE_SETUP="$LIO_WS/install_humble_clean/setup.bash"
LIO_SETUP="$LIO_WS/install_official_system/setup.bash"
PYTHON="/home/hyz/anaconda3/envs/himloco/bin/python"
NET="enp3s0"
SOCKET="/tmp/sea_nav_shadow.sock"
RUN_ROOT="$ROOT/logs/go2_seanav_navigation/$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"
FIRST_PERSON_DIR="$ROOT/first_person_view"
FRONT_CAMERA_TOPIC="${SEA_NAV_FRONT_CAMERA_TOPIC:-/frontvideostream}"
FRONT_CAMERA_FPS="${SEA_NAV_FRONT_CAMERA_FPS:-30}"
FRONT_CAMERA_SIZE="${SEA_NAV_FRONT_CAMERA_SIZE:-1280x720}"
FRONT_CAMERA_FILE=""

POLICY="$ROOT/models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt"
NAV_POLICY="$ROOT/artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt"
NAV_METADATA="$ROOT/artifacts/go2_onboard/sea_nav_policy_peer_model_2000.json"
GOAL_X=""
GOAL_Y=""
FRONT_GOAL_DISTANCE=""
FRONT_GOAL_FORWARD=""
FRONT_GOAL_LEFT=""
NAV_VX_MAX=inf
NAV_VX_MIN=0
NAV_VY_MAX=0
FIXED_SPORT_VX=""
GOAL_TOLERANCE=0.15
GOAL_SLOWDOWN_DISTANCE=0.50
ASSUME_CLEAR_LIDAR=0
ENABLE_OFFICIAL_MOTION=0
NON_INTERACTIVE=0
CLEANED=0
SUMMARY_WRITTEN=0

CPU_STATUS=FAIL
MCF_STATUS=FAIL
RAW_SENSOR_STATUS=FAIL
TRANSFORM_STATUS=NOT_USED_OFFICIAL
DESKEW_STATUS=OFFICIAL_NATIVE
POINT_LIO_STATUS=NOT_USED_OFFICIAL
NAVIGATION_STATUS=NOT_STARTED
MOTION_STATUS=NOT_MEASURED
EXIT_REASON=NOT_RECORDED

# Keep redirected logs grep-friendly: colors are enabled only for a terminal.
if [[ -t 1 ]]; then
  RED=$'\033[31m'
  YELLOW=$'\033[33m'
  GREEN=$'\033[32m'
  BOLD=$'\033[1m'
  RESET=$'\033[0m'
else
  RED=''
  YELLOW=''
  GREEN=''
  BOLD=''
  RESET=''
fi

mkdir -p "$LOG_ROOT" "$PID_ROOT"

say() { printf '[GO2] %s\n' "$*"; }
say_red() { printf '%s[GO2] %s%s\n' "$RED" "$*" "$RESET"; }
say_yellow() { printf '%s[GO2] %s%s\n' "$YELLOW" "$*" "$RESET"; }
say_green() { printf '%s[GO2] %s%s\n' "$GREEN" "$*" "$RESET"; }
banner_red() { printf '%s%s========== %s ==========%s\n' "$RED" "$BOLD" "$*" "$RESET"; }
pass() { printf '[PASS] %s\n' "$*"; }
fail() {
  if [[ -t 2 ]]; then
    printf '%s[FAIL] %s%s\n' "$RED" "$*" "$RESET" >&2
  else
    printf '[FAIL] %s\n' "$*" >&2
  fi
}
die() { fail "$*"; exit 1; }

source_ros() {
  set +u
  source "$ROS_SETUP"
  source "$UNITREE_SETUP"
  source "$LIO_BASE_SETUP"
  source "$LIO_SETUP"
  set -u
}

write_pid() { printf '%s\n' "$2" >"$PID_ROOT/$1.pid"; }
pid_alive() { kill -0 "$1" 2>/dev/null; }

wait_dead() {
  local pid="$1" deadline=$((SECONDS + 6))
  while pid_alive "$pid" && ((SECONDS < deadline)); do sleep 0.2; done
  ! pid_alive "$pid"
}

stop_pid() {
  local label="$1" file="$2" signal="${3:-INT}" pid
  [[ -f "$file" ]] || return 0
  pid="$(<"$file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  pid_alive "$pid" || return 0
  say "[STOP] $label pid=$pid signal=$signal"
  kill -"$signal" "$pid" 2>/dev/null || true
  if ! wait_dead "$pid"; then
    kill -TERM "$pid" 2>/dev/null || true
    if ! wait_dead "$pid"; then
      fail "$label did not stop gracefully; escalating to SIGKILL pid=$pid"
      kill -KILL "$pid" 2>/dev/null || true
      wait_dead "$pid" || true
    fi
  fi
  pid_alive "$pid" && fail "${label}_STOP=FAIL" || pass "${label}_STOP=PASS"
}

ancestor_pids() {
  local pid="$$" parent
  while [[ "$pid" =~ ^[0-9]+$ ]] && ((pid > 1)); do
    printf '%s\n' "$pid"
    parent="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ "$parent" =~ ^[0-9]+$ ]] || break
    [[ "$parent" == "$pid" ]] && break
    pid="$parent"
  done
}

kill_project_processes() {
  local pattern="$1" pid cmd ancestors
  ancestors="$(ancestor_pids | tr '\n' ' ')"
  while read -r pid cmd; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ " $ancestors " == *" $pid "* ]] && continue
    [[ "$cmd" == *"go2_seanav_navigation_test.sh"* ]] && continue
    kill -KILL "$pid" 2>/dev/null || true
  done < <(pgrep -af "$pattern" || true)
}

cleanup_old_processes() {
  say_red "[0] CLEAN OLD PROCESSES"
  kill_project_processes 'sea_nav|pointlio|point_lio|deskew_node.py|transform_everything|sensor_bridge|go2_shadow|himloco'
  sleep 5
  local residual
  residual="$(pgrep -af 'sea_nav|pointlio|point_lio|deskew_node.py|transform_everything|sensor_bridge|go2_shadow|himloco' || true)"
  if [[ -n "$residual" ]]; then
    printf '%s\n' "$residual"
    say "[CLEANUP] residual process exists"
    kill_project_processes 'sea_nav|pointlio|point_lio|deskew_node.py|transform_everything|sensor_bridge|go2_shadow|himloco'
    sleep 2
  fi
  ros2 daemon stop >/dev/null 2>&1 || true
  ros2 daemon start >/dev/null 2>&1 || true
  sleep 2
  say_red "PROCESS_CLEAN=PASS"
}

topic_rate() {
  local topic="$1" sample_seconds="${2:-6}" output rate
  output="$(timeout "${sample_seconds}s" ros2 topic hz "$topic" 2>&1 || true)"
  rate="$(printf '%s\n' "$output" | awk '/average rate:/ {print $3}' | tail -1)"
  [[ "$rate" =~ ^[0-9]+([.][0-9]+)?$ ]] && printf '%s\n' "$rate" || printf '0\n'
}

check_rate() {
  local topic="$1" minimum="$2" sample_seconds="${3:-6}" rate
  rate="$(topic_rate "$topic" "$sample_seconds")"
  printf '[GO2] %s rate=%sHz required>=%sHz\n' "$topic" "$rate" "$minimum"
  awk -v rate="$rate" -v min="$minimum" 'BEGIN {exit !(rate >= min)}'
}

check_topic_ready_with_retry() {
  local topic="$1" minimum="$2" failure="$3"
  local deadline=$((SECONDS + 30))
  while ((SECONDS < deadline)); do
    say_yellow "[WAIT] waiting $topic"
    if check_rate "$topic" "$minimum" 2; then
      say_green "[READY] $topic"
      return 0
    fi
  done
  die "$failure"
}

check_cpu() {
  say_red "[CPU] checking performance governor"
  command -v cpupower >/dev/null 2>&1 || die "CPU_NOT_READY: cpupower missing"
  cpupower frequency-info 2>&1 | tee "$LOG_ROOT/cpupower.log" || true
  local bad=0 file governor
  shopt -s nullglob
  local policies=(/sys/devices/system/cpu/cpufreq/policy*/scaling_governor)
  shopt -u nullglob
  ((${#policies[@]})) || bad=1
  for file in "${policies[@]}"; do
    governor="$(<"$file")"
    [[ "$governor" == performance ]] || bad=1
  done
  if ((bad)); then
    printf '[FAIL] CPU governor is not performance\n' >&2
    printf 'sudo cpupower frequency-set -g performance\n' >&2
    say_red "CPU_NOT_READY"
    die "CPU governor is not performance"
  fi
  CPU_STATUS=PASS
  say_red "CPU=performance"
}

check_mcf() {
  local output
  [[ -x "$PYTHON" ]] || die "MCF_STATUS=FAIL: missing SDK Python"
  if ! output="$($PYTHON - "$NET" <<'PY'
import sys
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

try:
    ChannelFactoryInitialize(0, sys.argv[1])
    client = MotionSwitcherClient()
    client.SetTimeout(5.0)
    client.Init()
    code, data = client.CheckMode()
    if code != 0:
        print(f"INITIAL_MODE_CHECK=FAIL:{code}")
        raise SystemExit(2)
    initial = str((data or {}).get("name", "") or "")
    print(f"INITIAL_MODE={initial}")
    # Official SportClient/MPC owns the motion service.  Releasing MCF here
    # would hand control to the low-level path and break the official backend.
    print("RELEASE_REQUEST=NOT_REQUESTED")
    final_code, final_data = client.CheckMode()
    final = str((final_data or {}).get("name", "") or "")
    print(f"FINAL_MODE={final}")
    print("MCF_STATUS=READ_ONLY" if final_code == 0 else "MCF_STATUS=FAIL")
    raise SystemExit(0 if final_code == 0 else 4)
except SystemExit:
    raise
except Exception as exc:
    print(f"MCF_EXCEPTION={type(exc).__name__}:{exc}")
    raise SystemExit(5)
PY
  )"; then
    printf '%s\n' "$output" | tee "$LOG_ROOT/mcf.log"
    die "MCF_STATUS=FAIL"
  fi
  printf '%s\n' "$output" | tee "$LOG_ROOT/mcf.log"
  grep -q '^MCF_STATUS=READ_ONLY$' "$LOG_ROOT/mcf.log" || die "MCF_STATUS=FAIL read-only check failed"
  MCF_STATUS=PASS
  say_red "MCF_STATUS=PASS"
}

start_sensor_bridge() {
  say_red "[START] sensor_bridge"
  local -a args=(
    /usr/bin/python3 -m deploy.go2_onboard.sensor_bridge
    --duration 0 --socket "$SOCKET" --log "$RUN_ROOT/sensor_bridge.jsonl" \
    --lowstate-topic /lowstate \
    --lidar-topic /utlidar/cloud_base \
    --odom-topic /utlidar/robot_odom \
    --wireless-topic /wirelesscontroller \
    --goal-topic /sea_nav/goal2d
  )
  [[ -n "$GOAL_X" ]] && args+=(--goal-x "$GOAL_X" --goal-y "$GOAL_Y")
  [[ -n "$FRONT_GOAL_DISTANCE" ]] && args+=(--front-goal-distance "$FRONT_GOAL_DISTANCE")
  [[ -n "$FRONT_GOAL_FORWARD" ]] && args+=(--front-goal-forward "$FRONT_GOAL_FORWARD" --front-goal-left "$FRONT_GOAL_LEFT")
  "${args[@]}" >"$LOG_ROOT/sensor_bridge.log" 2>&1 &
  write_pid sensor_bridge "$!"
  sleep 2
  pid_alive "$(<"$PID_ROOT/sensor_bridge.pid")" || die "sensor_bridge exited"
  check_rate /utlidar/cloud 10 || die "RAW_SENSOR=FAIL cloud"
  check_rate /utlidar/imu 100 || die "RAW_SENSOR=FAIL imu"
  RAW_SENSOR_STATUS=PASS
  pass "RAW_SENSOR=PASS"
}

find_descendant() {
  local root="$1" needle="$2" child cmd
  for child in $(pgrep -P "$root" 2>/dev/null || true); do
    cmd="$(ps -o args= -p "$child" 2>/dev/null || true)"
    [[ "$cmd" == *"$needle"* ]] && { printf '%s\n' "$child"; return 0; }
    find_descendant "$child" "$needle" && return 0
  done
  return 1
}

check_official_sensor_chain() {
  say_red "[CHECK] Unitree official LiDAR localization"
  check_topic_ready_with_retry /utlidar/cloud_base 10 CLOUD_BASE_NOT_READY
  check_topic_ready_with_retry /utlidar/robot_odom 10 ROBOT_ODOM_NOT_READY
  timeout 5s ros2 topic echo /utlidar/robot_odom --once >"$LOG_ROOT/official_odom_once.log" 2>&1 \
    || die "ROBOT_ODOM_NOT_READY"
  grep -q 'frame_id: odom' "$LOG_ROOT/official_odom_once.log" \
    || die "ROBOT_ODOM_NOT_READY frame_id"
  grep -q 'child_frame_id: base_link' "$LOG_ROOT/official_odom_once.log" \
    || die "ROBOT_ODOM_NOT_READY child_frame_id"
  check_rate /utlidar/cloud_deskewed 10 || die "OFFICIAL_LIDAR=FAIL cloud_deskewed"
  TRANSFORM_STATUS=OFFICIAL_NATIVE
  DESKEW_STATUS=OFFICIAL_NATIVE
  POINT_LIO_STATUS=NOT_USED_OFFICIAL
  say_red "OFFICIAL_LIDAR_CHAIN=PASS"
}

start_monitors() {
  say_red "[MONITOR] official cloud / deskew / odom"
  ros2 topic hz /utlidar/cloud >"$LOG_ROOT/cloud_hz.log" 2>&1 &
  write_pid cloud_monitor "$!"
  ros2 topic hz /utlidar/cloud_deskewed >"$LOG_ROOT/deskew_hz.log" 2>&1 &
  write_pid deskew_monitor "$!"
  ros2 topic hz /utlidar/robot_odom >"$LOG_ROOT/odom_hz.log" 2>&1 &
  write_pid odom_monitor "$!"
  ros2 topic echo /utlidar/robot_odom --qos-reliability reliable >"$RUN_ROOT/odom.jsonl" 2>&1 &
  write_pid odom_echo "$!"
  /usr/bin/python3 -m deploy.go2_onboard.diagnostics \
    --duration 0 \
    --report-interval 0.5 \
    --max-sensor-age 0.25 \
    --lowstate-topic /lowstate \
    --lidar-topic /utlidar/cloud_base \
    --odom-topic /utlidar/robot_odom \
    --wireless-topic /wirelesscontroller \
    --sport-state-topic rt/sportmodestate \
    --goal-topic /sea_nav/goal2d \
    --log "$RUN_ROOT/state_diagnostics.jsonl" \
    >"$LOG_ROOT/state_diagnostics.stdout" 2>&1 &
  write_pid state_diagnostics "$!"
}

start_navigation() {
  [[ -f "$NAV_POLICY" ]] || die "missing navigation policy: $NAV_POLICY"
  [[ -f "$NAV_METADATA" ]] || die "missing navigation metadata: $NAV_METADATA"
  say_yellow "WAIT_FOR_POSE_ARM: official Sport/MPC path; motion requires --enable-official-motion"
  local -a args=(
    "$NET"
    --sensor-socket "$SOCKET"
    --navigation-policy "$NAV_POLICY"
    --navigation-metadata "$NAV_METADATA"
    --navigation-vx-max "$NAV_VX_MAX"
    --navigation-vx-min "$NAV_VX_MIN"
    --navigation-vy-max "$NAV_VY_MAX"
    --goal-tolerance "$GOAL_TOLERANCE"
    --goal-slowdown-distance "$GOAL_SLOWDOWN_DISTANCE"
    --navigation-log "$RUN_ROOT/navigation.log"
  )
  [[ -n "$GOAL_X" ]] && args+=(--goal-x "$GOAL_X" --goal-y "$GOAL_Y")
  [[ -n "$FIXED_SPORT_VX" ]] && args+=(--fixed-sport-vx "$FIXED_SPORT_VX")
  ((ASSUME_CLEAR_LIDAR)) && args+=(--assume-clear-lidar)
  ((ENABLE_OFFICIAL_MOTION)) && args+=(--enable-motion)
  say_red "[START] SEA-Nav navigation + Unitree official Sport/MPC"
  if ((ENABLE_OFFICIAL_MOTION)); then
    say_red "WAIT_FOR_POSE_ARM: type START only after the robot is clear and already safely standing"
    if ((NON_INTERACTIVE)); then
      say_red "NON_INTERACTIVE=YES: proceeding with official SportClient.Move"
    else
      read -r -p "Type START to enable SportClient.Move, or Ctrl+C to abort: " confirmation
      [[ "$confirmation" == START ]] || die "official motion not confirmed"
    fi
  fi
  "$PYTHON" -m deploy.go2_onboard.sea_nav_sport_navigation \
    "${args[@]}" >"$RUN_ROOT/himloco.log" 2>&1 &
  write_pid navigation "$!"
  NAVIGATION_STATUS=RUNNING
}

start_front_camera_recording() {
  [[ -f "$ROOT/tools/go2_front_camera_recorder.py" ]] || die "FRONT_CAMERA=FAIL: recorder missing"
  mkdir -p "$FIRST_PERSON_DIR"
  local stamp candidate suffix=1
  stamp="$(date +%Y%m%d_%H%M)"
  candidate="$FIRST_PERSON_DIR/first_person_view_${stamp}.mp4"
  while [[ -e "$candidate" ]]; do
    candidate="$FIRST_PERSON_DIR/first_person_view_${stamp}_$(printf '%02d' "$suffix").mp4"
    suffix=$((suffix + 1))
  done
  FRONT_CAMERA_FILE="$candidate"
  say_red "[START] Go2 head front camera recording: $FRONT_CAMERA_TOPIC -> $FRONT_CAMERA_FILE"
  /usr/bin/python3 "$ROOT/tools/go2_front_camera_recorder.py" \
    --topic "$FRONT_CAMERA_TOPIC" --output "$FRONT_CAMERA_FILE" \
    --fps "$FRONT_CAMERA_FPS" --size "$FRONT_CAMERA_SIZE" \
    >"$LOG_ROOT/front_camera.log" 2>&1 &
  write_pid front_camera "$!"
  local deadline=$((SECONDS + 12))
  while ((SECONDS < deadline)); do
    grep -Eq 'first (encoded )?frame received; (MP4|H264) recording started' "$LOG_ROOT/front_camera.log" && break
    pid_alive "$(<"$PID_ROOT/front_camera.pid")" || die "FRONT_CAMERA=FAIL: recorder exited; see $LOG_ROOT/front_camera.log"
    sleep 0.5
  done
  grep -Eq 'first (encoded )?frame received; (MP4|H264) recording started' "$LOG_ROOT/front_camera.log" \
    || die "FRONT_CAMERA=FAIL: no head-camera frames received; see $LOG_ROOT/front_camera.log"
  pass "FRONT_CAMERA=RECORDING"
}

record_exit_reason() {
  [[ -f "$RUN_ROOT/himloco.log" ]] || return 0
  if grep -Eq 'error:|Traceback \(most recent call last\)' "$RUN_ROOT/himloco.log"; then
    EXIT_REASON=PROCESS_ERROR
    NAVIGATION_STATUS=FAIL
  elif grep -Fq '[safety] B' "$RUN_ROOT/himloco.log"; then
    EXIT_REASON=B_ESTOP
  elif grep -Fq '[safety] Ctrl+C' "$RUN_ROOT/himloco.log"; then
    EXIT_REASON=CTRL_C
  elif grep -Fq '[safety] refusing/stopping' "$RUN_ROOT/himloco.log"; then
    EXIT_REASON=REFUSING_OR_STOPPING
  elif [[ "$NAVIGATION_STATUS" == ENDED ]]; then
    EXIT_REASON=PROCESS_EXIT
  fi
}

write_motion_metrics() {
  [[ -f "$RUN_ROOT/odom.jsonl" ]] || return 0
  /usr/bin/python3 - "$RUN_ROOT/odom.jsonl" <<'PY' >"$LOG_ROOT/motion_metrics.log"
import math, re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
n = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
stamps = [float(a) + float(b) * 1e-9 for a, b in re.findall(rf"stamp:\s*\n\s*sec:\s*(-?\d+)\s*\n\s*nanosec:\s*(\d+)", text)]
positions = [tuple(float(v) for v in m) for m in re.findall(rf"position:\s*\n\s*x:\s*({n})\s*\n\s*y:\s*({n})\s*\n\s*z:\s*({n})", text)]
velocities = [tuple(float(v) for v in m) for m in re.findall(rf"linear:\s*\n\s*x:\s*({n})\s*\n\s*y:\s*({n})\s*\n\s*z:\s*({n})", text)]
print(f"ODOM_SAMPLES={len(positions)}")
if len(positions) < 2:
    print("MOTION_ODOM=FAIL")
    raise SystemExit(2)
start, end = positions[0], positions[-1]
disp = math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3)))
path = sum(math.sqrt(sum((positions[i][a] - positions[i-1][a]) ** 2 for a in range(3))) for i in range(1, len(positions)))
duration = stamps[-1] - stamps[0] if len(stamps) >= 2 else float("nan")
speeds = [math.sqrt(sum(x*x for x in v)) for v in velocities]
print(f"ODOM_DISPLACEMENT_M={disp:.6f}")
print(f"ODOM_PATH_DISTANCE_M={path:.6f}")
print(f"MOTION_TIME_S={duration:.6f}")
print(f"ODOM_FINAL_SPEED_MPS={speeds[-1]:.6f}" if speeds else "ODOM_FINAL_SPEED_MPS=nan")
print(f"ODOM_AVG_SPEED_MPS={sum(speeds)/len(speeds):.6f}" if speeds else "ODOM_AVG_SPEED_MPS=nan")
print("MOTION_ODOM=PASS")
PY
}

write_summary() {
  ((SUMMARY_WRITTEN)) && return 0
  SUMMARY_WRITTEN=1
  local policy_sha nav_sha cloud_rate deskew_rate odom_rate
  policy_sha="$(sha256sum "$POLICY" 2>/dev/null | awk '{print $1}' || true)"
  nav_sha="$(sha256sum "$NAV_POLICY" 2>/dev/null | awk '{print $1}' || true)"
  cloud_rate="$(awk '/average rate:/ {v=$3} END {print v+0}' "$LOG_ROOT/cloud_hz.log" 2>/dev/null || printf '0')"
  deskew_rate="$(awk '/average rate:/ {v=$3} END {print v+0}' "$LOG_ROOT/deskew_hz.log" 2>/dev/null || printf '0')"
  odom_rate="$(awk '/average rate:/ {v=$3} END {print v+0}' "$LOG_ROOT/odom_hz.log" 2>/dev/null || printf '0')"
  write_motion_metrics || true
  {
    printf 'CPU_STATUS=%s\nMCF_STATUS=%s\nRAW_SENSOR_STATUS=%s\n' "$CPU_STATUS" "$MCF_STATUS" "$RAW_SENSOR_STATUS"
    printf 'TRANSFORM_STATUS=%s\nDESKEW_STATUS=%s\nPOINT_LIO_STATUS=%s\n' "$TRANSFORM_STATUS" "$DESKEW_STATUS" "$POINT_LIO_STATUS"
    printf 'LOCOMOTION_BACKEND=UNITREE_SPORT_MPC\nOFFICIAL_MOTION_ENABLED=%s\n' "$ENABLE_OFFICIAL_MOTION"
    if [[ -n "$FIXED_SPORT_VX" ]]; then
      printf 'COMMAND_SOURCE=FIXED_SPORT_DIAGNOSTIC\nFIXED_SPORT_VX=%s\n' "$FIXED_SPORT_VX"
    else
      printf 'COMMAND_SOURCE=SEA_NAVIGATION\n'
    fi
  printf 'POLICY_UNUSED_BY_SPORT_MPC=%s\nPOLICY_SHA256=%s\n' "$POLICY" "$policy_sha"
    printf 'NAVIGATION_POLICY=%s\nNAVIGATION_POLICY_SHA256=%s\nNAVIGATION_METADATA=%s\n' "$NAV_POLICY" "$nav_sha" "$NAV_METADATA"
    printf 'GOAL_X=%s\nGOAL_Y=%s\nFRONT_GOAL_DISTANCE=%s\nGOAL_TOLERANCE=%s\nNAVIGATION_VX_MIN=%s\nNAVIGATION_VX_MAX=%s\nNAVIGATION_VY_MAX=%s\nASSUME_CLEAR_LIDAR=%s\n' "$GOAL_X" "$GOAL_Y" "$FRONT_GOAL_DISTANCE" "$GOAL_TOLERANCE" "$NAV_VX_MIN" "$NAV_VX_MAX" "$NAV_VY_MAX" "$ASSUME_CLEAR_LIDAR"
    printf 'LIDAR_RATE_HZ=%s\nOFFICIAL_DESKEW_RATE_HZ=%s\nROBOT_ODOM_RATE_HZ=%s\n' "$cloud_rate" "$deskew_rate" "$odom_rate"
    printf 'STATE_DIAGNOSTICS=%s\n' "$RUN_ROOT/state_diagnostics.jsonl"
    printf 'FIRST_PERSON_VIDEO=%s\nFRONT_CAMERA_TOPIC=%s\n' "$FRONT_CAMERA_FILE" "$FRONT_CAMERA_TOPIC"
    [[ -f "$LOG_ROOT/motion_metrics.log" ]] && cat "$LOG_ROOT/motion_metrics.log" || printf 'MOTION_ODOM=NOT_AVAILABLE\n'
    printf 'NAVIGATION_STATUS=%s\nMOTION_STATUS=%s\n' "$NAVIGATION_STATUS" "$MOTION_STATUS"
    printf 'EXIT_REASON=%s\n' "$EXIT_REASON"
    if [[ "$CPU_STATUS" == PASS && "$MCF_STATUS" == PASS && "$RAW_SENSOR_STATUS" == PASS && "$TRANSFORM_STATUS" == OFFICIAL_NATIVE && "$DESKEW_STATUS" == OFFICIAL_NATIVE && "$POINT_LIO_STATUS" == NOT_USED_OFFICIAL && "$NAVIGATION_STATUS" != NOT_STARTED && "$NAVIGATION_STATUS" != FAIL ]]; then
      printf 'RESULT=PASS\n'
    else
      printf 'RESULT=FAIL\n'
    fi
  } >"$RUN_ROOT/summary.txt"
}

cleanup() {
  local rc=$?
  ((CLEANED)) && return "$rc"
  CLEANED=1
  stop_pid ODOM_ECHO "$PID_ROOT/odom_echo.pid" INT || true
  stop_pid STATE_DIAGNOSTICS "$PID_ROOT/state_diagnostics.pid" INT || true
  stop_pid ODOM_MONITOR "$PID_ROOT/odom_monitor.pid" INT || true
  stop_pid DESKEW_MONITOR "$PID_ROOT/deskew_monitor.pid" INT || true
  stop_pid CLOUD_MONITOR "$PID_ROOT/cloud_monitor.pid" INT || true
  stop_pid NAVIGATION "$PID_ROOT/navigation.pid" INT || true
  stop_pid FRONT_CAMERA "$PID_ROOT/front_camera.pid" INT || true
  stop_pid ODOM_ADAPTER "$PID_ROOT/adapter.pid" INT || true
  stop_pid POINT_LIO "$PID_ROOT/pointlio.pid" INT || true
  stop_pid TRANSFORM "$PID_ROOT/transform.pid" INT || true
  stop_pid LIO_LAUNCH "$PID_ROOT/launch.pid" INT || true
  stop_pid SENSOR_BRIDGE "$PID_ROOT/sensor_bridge.pid" TERM || true
  record_exit_reason
  say_red "EXIT_REASON=$EXIT_REASON"
  write_summary || true
  printf '\n'
  banner_red "SEA-NAV GO2 NAVIGATION SUMMARY"
  printf 'CPU=%s MCF=%s SENSOR=%s TRANSFORM=%s DESKEW=%s POINT_LIO=%s NAVIGATION=%s MOTION=%s\n' \
    "$CPU_STATUS" "$MCF_STATUS" "$RAW_SENSOR_STATUS" "$TRANSFORM_STATUS" "$DESKEW_STATUS" "$POINT_LIO_STATUS" "$NAVIGATION_STATUS" "$MOTION_STATUS"
  printf 'SUMMARY=%s\nRUN_ROOT=%s\n' "$RUN_ROOT/summary.txt" "$RUN_ROOT"
  return "$rc"
}

usage() {
  cat <<'EOF'
Usage: bash tools/go2_seanav_navigation_test.sh --goal-x X --goal-y Y [options]
  --policy PATH              HIMLoco 1460 policy
  --navigation-policy PATH   SEA-Nav 550->3 policy
  --navigation-metadata PATH SEA-Nav policy metadata
  --goal-x VALUE             world-frame goal x
  --goal-y VALUE             world-frame goal y
  --front-goal-distance M    fix a goal M meters ahead of first odom pose
  --front-goal-forward M     fix a goal M meters forward from first pose
  --front-goal-left M        fix a goal M meters left from first pose
  --navigation-vx-max VALUE  forward speed limit, default inf (disabled)
  --navigation-vx-min VALUE  reverse speed floor, default 0 (disabled)
  --navigation-vy-max VALUE  lateral speed limit, default 0
  --front-camera-topic TOPIC  Go2 head front camera topic, default /frontvideostream
  --front-camera-fps VALUE    front camera capture rate, default 30
  --front-camera-size VALUE   V4L2 capture size, default 1280x720
  --fixed-sport-vx VALUE      diagnostic fixed SportClient command, no speed cap
  --goal-tolerance VALUE     goal stop radius, default 0.15m
  --goal-slowdown-distance M  start smooth forward braking, default 0.50m
  --assume-clear-lidar       explicit no-obstacle-avoidance test mode
  --enable-official-motion   explicitly forward SEA-Nav commands to SportClient.Move
  --non-interactive           skip the text START confirmation (for one-click wrapper)
EOF
}

handle_signal() { EXIT_REASON=CTRL_C; cleanup; exit 130; }
trap cleanup EXIT
trap handle_signal INT TERM

main() {
  while (($#)); do
    case "$1" in
      --policy) (($# >= 2)) || die "--policy requires a value"; POLICY="$2"; shift 2 ;;
      --navigation-policy) (($# >= 2)) || die "--navigation-policy requires a value"; NAV_POLICY="$2"; shift 2 ;;
      --navigation-metadata) (($# >= 2)) || die "--navigation-metadata requires a value"; NAV_METADATA="$2"; shift 2 ;;
      --goal-x) (($# >= 2)) || die "--goal-x requires a value"; GOAL_X="$2"; shift 2 ;;
      --goal-y) (($# >= 2)) || die "--goal-y requires a value"; GOAL_Y="$2"; shift 2 ;;
      --front-goal-distance) (($# >= 2)) || die "--front-goal-distance requires a value"; FRONT_GOAL_DISTANCE="$2"; shift 2 ;;
      --front-goal-forward) (($# >= 2)) || die "--front-goal-forward requires a value"; FRONT_GOAL_FORWARD="$2"; shift 2 ;;
      --front-goal-left) (($# >= 2)) || die "--front-goal-left requires a value"; FRONT_GOAL_LEFT="$2"; shift 2 ;;
      --navigation-vx-max) (($# >= 2)) || die "--navigation-vx-max requires a value"; NAV_VX_MAX="$2"; shift 2 ;;
      --navigation-vx-min) (($# >= 2)) || die "--navigation-vx-min requires a value"; NAV_VX_MIN="$2"; shift 2 ;;
      --navigation-vy-max) (($# >= 2)) || die "--navigation-vy-max requires a value"; NAV_VY_MAX="$2"; shift 2 ;;
      --front-camera-topic) (($# >= 2)) || die "--front-camera-topic requires a value"; FRONT_CAMERA_TOPIC="$2"; shift 2 ;;
      --front-camera-fps) (($# >= 2)) || die "--front-camera-fps requires a value"; FRONT_CAMERA_FPS="$2"; shift 2 ;;
      --front-camera-size) (($# >= 2)) || die "--front-camera-size requires a value"; FRONT_CAMERA_SIZE="$2"; shift 2 ;;
      --fixed-sport-vx) (($# >= 2)) || die "--fixed-sport-vx requires a value"; FIXED_SPORT_VX="$2"; shift 2 ;;
      --goal-tolerance) (($# >= 2)) || die "--goal-tolerance requires a value"; GOAL_TOLERANCE="$2"; shift 2 ;;
      --goal-slowdown-distance) (($# >= 2)) || die "--goal-slowdown-distance requires a value"; GOAL_SLOWDOWN_DISTANCE="$2"; shift 2 ;;
      --assume-clear-lidar) ASSUME_CLEAR_LIDAR=1; shift ;;
      --enable-official-motion) ENABLE_OFFICIAL_MOTION=1; shift ;;
      --non-interactive) NON_INTERACTIVE=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  if [[ -n "$FRONT_GOAL_DISTANCE" ]]; then
    [[ -z "$GOAL_X" && -z "$GOAL_Y" ]] || die "front goal cannot be combined with goal-x/goal-y"
    [[ -z "$FRONT_GOAL_FORWARD" && -z "$FRONT_GOAL_LEFT" ]] || die "front goal distance cannot be combined with offsets"
  elif [[ -n "$FRONT_GOAL_FORWARD" || -n "$FRONT_GOAL_LEFT" ]]; then
    [[ -z "$GOAL_X" && -z "$GOAL_Y" ]] || die "front goal cannot be combined with goal-x/goal-y"
    [[ -n "$FRONT_GOAL_FORWARD" && -n "$FRONT_GOAL_LEFT" ]] || die "front-goal-forward and --front-goal-left must be used together"
  else
    [[ -n "$GOAL_X" && -n "$GOAL_Y" ]] || die "provide --goal-x/--goal-y or --front-goal-distance"
  fi
  cd "$ROOT"
  source_ros
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
  cleanup_old_processes
  printf 'RMW=%s\nROS_DOMAIN_ID=%s\nROS_LOCALHOST_ONLY=%s\n' "$RMW_IMPLEMENTATION" "$ROS_DOMAIN_ID" "$ROS_LOCALHOST_ONLY"
  check_cpu
  check_mcf
  start_sensor_bridge
  check_official_sensor_chain
  start_front_camera_recording
  start_monitors
  start_navigation
  say_yellow "WAIT_FOR_POSE_ARM: manually Start, then A; press B for ESTOP"
  say_red "READY: sensor chain and SEA-Nav navigation are running"
  say_red "[EXPERIMENT] motion validation active; B or Ctrl+C ends the run"
  while pid_alive "$(<"$PID_ROOT/navigation.pid")"; do
    printf '\n%s========== GO2 STATUS ==========%s\n' "$BOLD" "$RESET"
    printf 'CPU=%s MCF=%s RAW_SENSOR=%s TRANSFORM=%s DESKEW=%s POINT_LIO=%s NAVIGATION=%s\n' \
      "$CPU_STATUS" "$MCF_STATUS" "$RAW_SENSOR_STATUS" "$TRANSFORM_STATUS" "$DESKEW_STATUS" "$POINT_LIO_STATUS" "${RED}${NAVIGATION_STATUS}${RESET}"
    sleep 5
  done
  NAVIGATION_STATUS=ENDED
  MOTION_STATUS=RECORDED
  record_exit_reason
  say_red "[EXPERIMENT] navigation process ended; cleanup will now run"
}

main "$@"

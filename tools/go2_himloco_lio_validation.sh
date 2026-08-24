#!/usr/bin/env bash
set -Eeuo pipefail

# Unified daily Go2 validation entry point.  Start/A remain manual; B remains
# the controller ESTOP.  The default HIMLoco policy is opt-out via
# --no-himloco.

ROOT="/home/hyz/桌面/sea_nav"
LIO_WS="$ROOT/lio/sea_nav_lio_ws"
ROS_SETUP="/opt/ros/humble/setup.bash"
UNITREE_SETUP="/home/hyz/unitree_msgs_humble_ws/install/setup.bash"
LIO_SETUP="$LIO_WS/install_humble_clean/setup.bash"
HIMLOCO_PYTHON="/home/hyz/anaconda3/envs/himloco/bin/python"
NET="enp3s0"
RUN_ROOT="$ROOT/logs/go2_himloco_lio_validation/$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"

POLICY="$ROOT/models/locomotion/himloco/himloco_himppo_continuous_turning_policy_1460.pt"
VX=0.1
VY=0
WZ=0
RUN_HIMLOCO=0
KEEP_ALIVE=0
CLEANED=0

CPU_STATUS=FAIL
MCF_STATUS=FAIL
RAW_SENSOR_STATUS=FAIL
TRANSFORM_STATUS=FAIL
DESKEW_STATUS=FAIL
POINT_LIO_STATUS=FAIL
HIMLOCO_STATUS=SKIPPED
MOTION_STATUS=SKIPPED
SUMMARY_WRITTEN=0

mkdir -p "$LOG_ROOT" "$PID_ROOT"

say() { printf '[GO2] %s\n' "$*"; }
pass() { printf '[PASS] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; }
die() { fail "$*"; exit 1; }

source_ros() {
  set +u
  source "$ROS_SETUP"
  source "$UNITREE_SETUP"
  source "$LIO_SETUP"
  set -u
}

write_pid() { printf '%s\n' "$2" >"$PID_ROOT/$1.pid"; }
pid_alive() { kill -0 "$1" 2>/dev/null; }

wait_dead() {
  local pid="$1" deadline=$((SECONDS + 5))
  while pid_alive "$pid" && ((SECONDS < deadline)); do sleep 0.2; done
  ! pid_alive "$pid"
}

stop_pid() {
  local label="$1" file="$2" signal="${3:-INT}" pid
  [[ -f "$file" ]] || return 0
  pid="$(<"$file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  pid_alive "$pid" || return 0
  say "[STOP] $label pid=$pid"
  kill -"$signal" "$pid" 2>/dev/null || true
  if ! wait_dead "$pid"; then
    kill -TERM "$pid" 2>/dev/null || true
    if ! wait_dead "$pid"; then
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
    [[ "$cmd" == *"go2_himloco_lio_validation.sh"* ]] && continue
    kill -KILL "$pid" 2>/dev/null || true
  done < <(pgrep -af "$pattern" || true)
}

cleanup_old_processes() {
  say "[0] CLEAN OLD PROCESSES"
  # Guarded equivalent of the requested project-specific pkill patterns.
  # It excludes this script and every shell ancestor.
  kill_project_processes 'sea_nav|pointlio|point_lio|deskew_node.py|transform_everything|sensor_bridge|go2_shadow|himloco'
  kill_project_processes 'ros2 launch|component_container'
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
  residual="$(ros2 node list 2>/dev/null | grep -E '^/(sea_nav_lidar_deskew|sea_nav_point_lio|sea_nav_sensor_transform|sea_nav_go2_shadow_runtime)$' || true)"
  [[ -z "$residual" ]] || say "[CLEANUP] warning: old ROS nodes remain\n$residual"
  pass "PROCESS_CLEAN=PASS"
}

cleanup() {
  local rc=$?
  ((CLEANED)) && return "$rc"
  CLEANED=1
  stop_pid CLOUD_MONITOR "$PID_ROOT/cloud_monitor.pid" INT || true
  stop_pid DESKEW_MONITOR "$PID_ROOT/deskew_monitor.pid" INT || true
  stop_pid ODOM_MONITOR "$PID_ROOT/odom_monitor.pid" INT || true
  stop_pid ODOM_ECHO "$PID_ROOT/odom_echo.pid" INT || true
  stop_pid HIMLOCO "$PID_ROOT/himloco.pid" INT || true
  stop_pid POINT_LIO "$PID_ROOT/pointlio.pid" INT || true
  stop_pid TRANSFORM "$PID_ROOT/transform.pid" INT || true
  stop_pid LIO_LAUNCH "$PID_ROOT/launch.pid" INT || true
  stop_pid SENSOR_BRIDGE "$PID_ROOT/sensor_bridge.pid" TERM || true
  write_summary || true
  printf '\n========== GO2 VALIDATION SUMMARY ==========\n'
  printf 'CPU: %s\n' "$CPU_STATUS"
  printf 'MCF: %s\n' "$MCF_STATUS"
  printf 'RAW_SENSOR: %s\n' "$RAW_SENSOR_STATUS"
  printf 'TRANSFORM: %s\n' "$TRANSFORM_STATUS"
  printf 'DESKEW: %s\n' "$DESKEW_STATUS"
  printf 'POINT_LIO: %s\n' "$POINT_LIO_STATUS"
  printf 'HIMLOCO: %s\n' "$HIMLOCO_STATUS"
  printf 'MOTION_20S: %s\n' "$MOTION_STATUS"
  printf 'CLEANUP=PASS\nSUMMARY=%s\nRUN_ROOT=%s\n' "$RUN_ROOT/summary.txt" "$RUN_ROOT"
  return "$rc"
}
handle_signal() {
  cleanup
  exit 130
}
trap cleanup EXIT
trap handle_signal INT TERM

check_cpu() {
  say "[CPU] cpupower frequency-info"
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
    fail "CPU_NOT_READY"
    printf 'sudo cpupower frequency-set -g performance\n' >&2
    die "CPU governor is not performance"
  fi
  CPU_STATUS=PASS
  pass "CPU=performance"
}

check_mcf() {
  local output
  [[ -x "$HIMLOCO_PYTHON" ]] || die "MCF_STATUS=FAIL: missing SDK Python"
  if ! output="$($HIMLOCO_PYTHON - "$NET" <<'PY'
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
    if initial == "mcf":
        print("RELEASE_REQUEST=REQUESTED")
        release_code, _ = client.ReleaseMode()
        if release_code != 0:
            print(f"RELEASE_RESULT=FAIL:{release_code}")
            raise SystemExit(3)
    else:
        print("MCF already released")
        print("RELEASE_REQUEST=NOT_NEEDED")
    final_code, final_data = client.CheckMode()
    final = str((final_data or {}).get("name", "") or "")
    print(f"FINAL_MODE={final}")
    print("RELEASE_RESULT=PASS" if final_code == 0 and final == "" else "RELEASE_RESULT=FAIL")
    raise SystemExit(0 if final_code == 0 and final == "" else 4)
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
  grep -q '^FINAL_MODE=$' "$LOG_ROOT/mcf.log" || die "MCF_STATUS=FAIL final mode is not empty"
  MCF_STATUS=PASS
  pass "MCF_STATUS=PASS"
}

topic_rate() {
  local output rate
  output="$(timeout 6s ros2 topic hz "$1" 2>&1 || true)"
  rate="$(printf '%s\n' "$output" | awk '/average rate:/ {print $3}' | tail -1)"
  [[ "$rate" =~ ^[0-9]+([.][0-9]+)?$ ]] && printf '%s\n' "$rate" || printf '0\n'
}

check_rate() {
  local rate
  rate="$(topic_rate "$1")"
  printf '[GO2] %s rate=%sHz required>=%sHz\n' "$1" "$rate" "$2"
  awk -v rate="$rate" -v min="$2" 'BEGIN {exit !(rate >= min)}'
}

start_sensor_bridge() {
  say "[START] sensor_bridge"
  /usr/bin/python3 -m deploy.go2_onboard.sensor_bridge \
    --duration 0 --log "$LOG_ROOT/sensor_bridge.jsonl" \
    --lowstate-topic /lowstate --lidar-topic /utlidar/cloud \
    --odom-topic /sea_nav/lio/odom \
    >"$LOG_ROOT/sensor_bridge.log" 2>&1 &
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

start_lio() {
  say "[START] point_lio_go2.launch.py deskew:=true"
  ros2 launch sea_nav_lio_bringup point_lio_go2.launch.py deskew:=true \
    >"$LOG_ROOT/point_lio.log" 2>&1 &
  local launch_pid="$!" transform_pid point_pid
  write_pid launch "$launch_pid"
  sleep 5
  pid_alive "$launch_pid" || die "LIO launch exited"
  transform_pid="$(find_descendant "$launch_pid" transform_everything || true)"
  point_pid="$(find_descendant "$launch_pid" pointlio_mapping || true)"
  [[ "$transform_pid" =~ ^[0-9]+$ ]] || die "transform process missing"
  [[ "$point_pid" =~ ^[0-9]+$ ]] || die "pointlio process missing"
  write_pid transform "$transform_pid"
  write_pid pointlio "$point_pid"
  ros2 node list 2>/dev/null | grep -q '^/sea_nav_sensor_transform$' || die "transform node missing"
  ros2 node list 2>/dev/null | grep -q '^/sea_nav_lidar_deskew$' || die "deskew node missing"
  ros2 node list 2>/dev/null | grep -q '^/sea_nav_point_lio$' || die "Point-LIO node missing"
  check_rate /sea_nav/lio/transformed_imu 100 || die "TRANSFORM=FAIL transformed IMU"
  check_rate /sea_nav/lio/deskewed_cloud 10 || die "DESKEW=FAIL deskew cloud"
  grep 'DESKEW_STATUS' "$LOG_ROOT/point_lio.log" | tail -3 || true
  grep -q 'enabled=true' "$LOG_ROOT/point_lio.log" || die "DESKEW=FAIL enabled status missing"
  TRANSFORM_STATUS=PASS
  DESKEW_STATUS=PASS
  LIO_SENSOR_CHAIN=PASS
  check_rate /sea_nav/lio/odom 5 || die "POINT_LIO=FAIL odom rate"
  timeout 5s ros2 topic echo /sea_nav/lio/odom --once >"$LOG_ROOT/odom_once.log" 2>&1 || die "POINT_LIO=FAIL odom data"
  POINT_LIO_STATUS=PASS
  pass "POINT_LIO=PASS"
}

start_himloco() {
  if ((!RUN_HIMLOCO)); then
    HIMLOCO_STATUS=SKIPPED
    say "HIMLOCO skipped (use --run-himloco to enable)"
    return
  fi
  [[ -f "$POLICY" ]] || die "policy not found: $POLICY"
  "$HIMLOCO_PYTHON" -c 'import numpy, torch, unitree_sdk2py; import deploy.go2_onboard.himloco_fixed_control' \
    >"$LOG_ROOT/himloco_preflight.log" 2>&1 || die "HIMLOCO import preflight failed"
  say "[START] HIMLoco policy=$POLICY vx=$VX vy=$VY wz=$WZ"
  "$HIMLOCO_PYTHON" -m deploy.go2_onboard.himloco_fixed_control \
    "$NET" --policy "$POLICY" --vx "$VX" --vy "$VY" --wz "$WZ" \
    --max-sensor-age 0.10 >"$LOG_ROOT/himloco.log" 2>&1 &
  write_pid himloco "$!"
  HIMLOCO_STATUS=STARTED_MANUAL_START_A
}

start_monitors() {
  say "[MONITOR] cloud/deskew/odom"
  ros2 topic hz /utlidar/cloud >"$LOG_ROOT/cloud_hz.log" 2>&1 &
  write_pid cloud_monitor "$!"
  ros2 topic hz /sea_nav/lio/deskewed_cloud >"$LOG_ROOT/deskew_hz.log" 2>&1 &
  write_pid deskew_monitor "$!"
  ros2 topic hz /sea_nav/lio/odom >"$LOG_ROOT/odom_hz.log" 2>&1 &
  write_pid odom_monitor "$!"
  ros2 topic echo /sea_nav/lio/odom --qos-reliability reliable >"$LOG_ROOT/odom_motion.log" 2>&1 &
  write_pid odom_echo "$!"
}

write_motion_metrics() {
  "$HIMLOCO_PYTHON" - "$LOG_ROOT/odom_motion.log" <<'PY' >"$LOG_ROOT/motion_metrics.log"
import math
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
number = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
stamps = [float(a) + float(b) * 1e-9 for a, b in re.findall(
    rf"stamp:\s*\n\s*sec:\s*(-?\d+)\s*\n\s*nanosec:\s*(\d+)", text, re.MULTILINE)]
positions = [tuple(float(v) for v in m) for m in re.findall(
    rf"position:\s*\n\s*x:\s*({number})\s*\n\s*y:\s*({number})\s*\n\s*z:\s*({number})", text, re.MULTILINE)]
velocities = [tuple(float(v) for v in m) for m in re.findall(
    rf"linear:\s*\n\s*x:\s*({number})\s*\n\s*y:\s*({number})\s*\n\s*z:\s*({number})", text, re.MULTILINE)]
print(f"ODOM_SAMPLES={len(positions)}")
if len(positions) < 2:
    print("MOTION_ODOM=FAIL")
    raise SystemExit(2)
start, end = positions[0], positions[-1]
displacement = math.sqrt(sum((end[i] - start[i]) ** 2 for i in range(3)))
path_distance = sum(math.sqrt(sum((positions[i][a] - positions[i - 1][a]) ** 2 for a in range(3))) for i in range(1, len(positions)))
duration = stamps[-1] - stamps[0] if len(stamps) >= 2 else float("nan")
print(f"ODOM_START_X={start[0]:.6f}")
print(f"ODOM_START_Y={start[1]:.6f}")
print(f"ODOM_END_X={end[0]:.6f}")
print(f"ODOM_END_Y={end[1]:.6f}")
print(f"ODOM_DISPLACEMENT_M={displacement:.6f}")
print(f"ODOM_PATH_DISTANCE_M={path_distance:.6f}")
print(f"MOTION_TIME_S={duration:.6f}")
if velocities:
    speeds = [math.sqrt(sum(v * v for v in sample)) for sample in velocities]
    print(f"ODOM_FINAL_SPEED_MPS={speeds[-1]:.6f}")
    print(f"ODOM_AVG_SPEED_MPS={sum(speeds) / len(speeds):.6f}")
else:
    print("ODOM_FINAL_SPEED_MPS=nan")
    print("ODOM_AVG_SPEED_MPS=nan")
print("MOTION_ODOM=PASS")
PY
}

wait_experiment() {
  ((RUN_HIMLOCO)) || return 0
  start_monitors
  say "[EXPERIMENT] press Start then A manually; B or Ctrl+C finishes"
  while pid_alive "$(<"$PID_ROOT/himloco.pid")"; do
    printf '\n========== GO2 STATUS ==========\n'
    printf 'CPU: %s\nMCF: %s\nRAW_SENSOR: %s\nTRANSFORM: %s\nDESKEW: %s\nPOINT_LIO: %s\nHIMLOCO: %s\n' \
      "$CPU_STATUS" "$MCF_STATUS" "$RAW_SENSOR_STATUS" "$TRANSFORM_STATUS" \
      "$DESKEW_STATUS" "$POINT_LIO_STATUS" "$HIMLOCO_STATUS"
    if grep -Eiq 'B[ _-]*ESTOP|ESTOP|\[safety\].*EXIT|SafetyError' "$LOG_ROOT/himloco.log"; then
      say "[EXPERIMENT] HIMLoco safety exit detected"
      break
    fi
    sleep 5
  done
  HIMLOCO_STATUS=ENDED
  write_motion_metrics || true
  if grep -q '^MOTION_ODOM=PASS$' "$LOG_ROOT/motion_metrics.log"; then
    MOTION_STATUS=PASS
    cat "$LOG_ROOT/motion_metrics.log"
  else
    MOTION_STATUS=FAIL
    say "[EXPERIMENT] odom motion metrics unavailable"
  fi
}

write_summary() {
  ((SUMMARY_WRITTEN)) && return 0
  SUMMARY_WRITTEN=1
  local policy_sha=UNAVAILABLE cloud_rate deskew_rate odom_rate
  [[ -f "$POLICY" ]] && policy_sha="$(sha256sum "$POLICY" | awk '{print $1}')"
  cloud_rate="$(awk '/average rate:/ {v=$3} END {print v+0}' "$LOG_ROOT/cloud_hz.log" 2>/dev/null || printf '0')"
  deskew_rate="$(awk '/average rate:/ {v=$3} END {print v+0}' "$LOG_ROOT/deskew_hz.log" 2>/dev/null || printf '0')"
  odom_rate="$(awk '/average rate:/ {v=$3} END {print v+0}' "$LOG_ROOT/odom_hz.log" 2>/dev/null || printf '0')"
  {
    printf 'CPU_STATUS=%s\nMCF_STATUS=%s\n' "$CPU_STATUS" "$MCF_STATUS"
    printf 'POLICY=%s\nPOLICY_SHA256=%s\n' "$POLICY" "$policy_sha"
    printf 'VX=%s\nVY=%s\nWZ=%s\n' "$VX" "$VY" "$WZ"
    printf 'LIDAR_RATE_HZ=%s\nDESKEW_RATE_HZ=%s\nODOM_RATE_HZ=%s\n' "$cloud_rate" "$deskew_rate" "$odom_rate"
    [[ -f "$LOG_ROOT/motion_metrics.log" ]] && cat "$LOG_ROOT/motion_metrics.log" || printf 'MOTION_ODOM=SKIPPED\n'
    printf 'RAW_SENSOR_STATUS=%s\nTRANSFORM_STATUS=%s\nDESKEW_STATUS=%s\n' "$RAW_SENSOR_STATUS" "$TRANSFORM_STATUS" "$DESKEW_STATUS"
    printf 'POINT_LIO_STATUS=%s\nHIMLOCO_STATUS=%s\nMOTION_STATUS=%s\n' "$POINT_LIO_STATUS" "$HIMLOCO_STATUS" "$MOTION_STATUS"
    if [[ "$RAW_SENSOR_STATUS" == PASS && "$TRANSFORM_STATUS" == PASS && "$DESKEW_STATUS" == PASS && "$POINT_LIO_STATUS" == PASS && ( "$HIMLOCO_STATUS" == SKIPPED || "$MOTION_STATUS" == PASS ) ]]; then
      printf 'RESULT=PASS\n'
    else
      printf 'RESULT=FAIL\n'
    fi
  } >"$RUN_ROOT/summary.txt"
}

usage() {
  cat <<'EOF'
Usage: bash tools/go2_himloco_lio_validation.sh [options]
  --policy PATH       HIMLoco policy
  --vx VALUE          forward command, default 0.1
  --vy VALUE          lateral command, default 0
  --wz VALUE          yaw command, default 0
  --run-himloco       run HIMLoco and perform the 20-second motion check
  --no-himloco        compatibility alias for the default LIO-only mode
  --keep-alive        keep checked background nodes after validation
EOF
}

main() {
  while (($#)); do
    case "$1" in
      --policy) (($# >= 2)) || die "--policy requires a value"; POLICY="$2"; shift 2 ;;
      --vx) (($# >= 2)) || die "--vx requires a value"; VX="$2"; shift 2 ;;
      --vy) (($# >= 2)) || die "--vy requires a value"; VY="$2"; shift 2 ;;
      --wz) (($# >= 2)) || die "--wz requires a value"; WZ="$2"; shift 2 ;;
      --run-himloco) RUN_HIMLOCO=1; shift ;;
      --no-himloco) RUN_HIMLOCO=0; shift ;;
      --keep-alive) KEEP_ALIVE=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  cd "$ROOT"
  source_ros
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
  cleanup_old_processes
  printf 'RMW=%s\nROS_DOMAIN_ID=%s\nROS_LOCALHOST_ONLY=%s\n' \
    "$RMW_IMPLEMENTATION" "$ROS_DOMAIN_ID" "$ROS_LOCALHOST_ONLY"
  check_cpu
  check_mcf
  start_sensor_bridge
  start_lio
  start_himloco
  wait_experiment
  say "READY: Start/A remain manual; B ESTOP remains active"
  if ((KEEP_ALIVE && !RUN_HIMLOCO)); then
    start_monitors
    while :; do sleep 5; done
  elif ((KEEP_ALIVE && RUN_HIMLOCO)); then
    while :; do sleep 5; done
  fi
}

main "$@"

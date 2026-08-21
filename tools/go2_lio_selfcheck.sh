#!/usr/bin/env bash
set -u

# Central supervisor for the read-only independent-LIO pipeline.
# This script never starts SEA-Nav/HIMLoco, creates LowCmd, sends motion
# commands, or presses Start/A. Its only active robot-side operation is the
# explicit MCF CheckMode -> conditional ReleaseMode -> CheckMode gate below.

REPO="/home/hyz/桌面/sea_nav"
LIO_WS="$REPO/lio/sea_nav_lio_ws"
CALIB="$REPO/logs/lio_odom_calibration/lio_reference_base_se2.yaml"
UNITREE_SETUP="/home/hyz/unitree_msgs_humble_ws/install/setup.bash"
LIO_SETUP="$LIO_WS/install_humble_clean/setup.bash"
NET="enp3s0"
PYTHON="/usr/bin/python3"
SDK_PYTHON="/home/hyz/anaconda3/envs/himloco/bin/python"
RMW="rmw_cyclonedds_cpp"
ROS_DOMAIN="0"
ROS_LOCALHOST="0"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGROOT="$REPO/logs/go2_lio_selfcheck/$STAMP"
STATUS_FILE="$LOGROOT/status.txt"
MCF_STATUS_FILE="$LOGROOT/mcf.status"
GOAL_STATUS="$LOGROOT/goal.status"

GOAL_X=""
GOAL_Y=""
FORWARD=""
GOAL_MODE=""
MANAGED=0
POINT_LIO_STARTED=0

POINT_LOG="$LOGROOT/point_lio.log"
ADAPTER_LOG="$LOGROOT/adapter.log"
BRIDGE_LOG="$LOGROOT/sensor_bridge.log"
DIAG_LOG="$LOGROOT/diagnostic.log"
POINT_STATUS="$LOGROOT/point_lio.status"
ADAPTER_STATUS="$LOGROOT/adapter.status"
BRIDGE_STATUS="$LOGROOT/sensor_bridge.status"
DIAG_STATUS="$LOGROOT/diagnostic.status"
POINT_PID="$LOGROOT/point_lio.pid"
ADAPTER_PID="$LOGROOT/adapter.pid"
BRIDGE_PID="$LOGROOT/sensor_bridge.pid"
DIAG_PID="$LOGROOT/diagnostic.pid"

CYCLONE="<CycloneDDS><Domain Id=\"any\"><General><Interfaces><NetworkInterface name=\"$NET\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"

mkdir -p "$LOGROOT"

say()  { printf '[SELFCHK] %s\n' "$*"; }
pass() { printf '[PASS] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; }

die() {
  fail "$*"
  printf 'LOG_DIR=%s\n' "$LOGROOT"
  exit 1
}

source_ros() {
  # ROS2 Humble setup files read AMENT_TRACE_SETUP_FILES under nounset.
  set +u
  source /opt/ros/humble/setup.bash
  source "$UNITREE_SETUP"
  set -u
}

export RMW_IMPLEMENTATION="$RMW"
export ROS_DOMAIN_ID="$ROS_DOMAIN"
export ROS_LOCALHOST_ONLY="$ROS_LOCALHOST"
export CYCLONEDDS_URI="$CYCLONE"
source_ros

quote() { printf '%q' "$1"; }

while (($#)); do
  case "$1" in
    --goal-x)
      (($# >= 2)) || die "missing value for --goal-x"
      GOAL_X="$2"; shift 2 ;;
    --goal-y)
      (($# >= 2)) || die "missing value for --goal-y"
      GOAL_Y="$2"; shift 2 ;;
    --forward)
      (($# >= 2)) || die "missing value for --forward"
      FORWARD="$2"; shift 2 ;;
    --managed)
      MANAGED=1; shift ;;
    *)
      die "unknown selfcheck argument: $1" ;;
  esac
done

if (( MANAGED )); then
  TERMINAL_EPILOGUE="exit"
else
  TERMINAL_EPILOGUE="exec bash"
fi

if [[ -n "$FORWARD" ]]; then
  [[ -z "$GOAL_X" && -z "$GOAL_Y" ]] || die "use --forward or --goal-x/--goal-y, not both"
  GOAL_MODE="FORWARD"
elif [[ -n "$GOAL_X" || -n "$GOAL_Y" ]]; then
  [[ -n "$GOAL_X" && -n "$GOAL_Y" ]] || die "both --goal-x and --goal-y are required"
  GOAL_MODE="WORLD"
fi

BRIDGE_GOAL_ARGS=""

write_goal_status() {
  printf 'GOAL_MODE=%s\nGOAL_X=%s\nGOAL_Y=%s\n' \
    "$GOAL_MODE" "$GOAL_X" "$GOAL_Y" >"$GOAL_STATUS"
}

set_bridge_goal() {
  if [[ -n "$GOAL_X" && -n "$GOAL_Y" ]]; then
    BRIDGE_GOAL_ARGS="--goal-x $(quote "$GOAL_X") --goal-y $(quote "$GOAL_Y")"
  fi
}

set_bridge_goal
[[ -z "$GOAL_X" || -z "$GOAL_Y" ]] || write_goal_status

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
for command_name in ros2 timeout awk grep ps ip gnome-terminal; do need_cmd "$command_name"; done

[[ -d "$LIO_WS" ]] || die "missing LIO workspace: $LIO_WS"
[[ -f "$LIO_SETUP" ]] || die "missing LIO setup: $LIO_SETUP"
[[ -f "$CALIB" ]] || die "missing calibration YAML: $CALIB"
[[ -f "$REPO/tools/go2_lio_diagnose.py" ]] || die "missing diagnostic Python"

topic_info() { ros2 topic info "$1" 2>/dev/null || true; }
publisher_count() { topic_info "$1" | awk '/Publisher count:/ {print $3; exit}'; }
topic_has_data() { timeout 4s ros2 topic echo "$1" --once >/dev/null 2>&1; }

check_imu_stream() {
  local topic="$1" label="$2" expected_frame="$3" tmp result
  local pubs
  pubs="$(publisher_count "$topic")"; [[ "$pubs" =~ ^[0-9]+$ ]] || pubs=0
  if (( pubs < 1 )); then
    say "${label}=FAIL"
    say "${label}_ACCEL_INVALID"
    say "POINT_LIO_START_BLOCKED"
    die "ROOT_CAUSE=${label}_ACCEL_INVALID (no publisher on $topic)"
  fi

  tmp="$(mktemp)"
  # timeout is expected after the short capture; validity comes from the data.
  timeout 4s ros2 topic echo "$topic" >"$tmp" 2>&1 || true
  if ! result="$($PYTHON - "$tmp" "$expected_frame" "$label" <<'PY'
import math
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
samples = []
current = None
section = None

def finish():
    if current is not None and all(k in current for k in (
            "sec", "nanosec", "gx", "gy", "gz", "ax", "ay", "az")):
        samples.append(dict(current))

for line in text.splitlines():
    stripped = line.strip()
    if stripped == "header:":
        finish()
        current = {}
        section = None
        continue
    if current is None:
        continue
    if stripped == "stamp:":
        section = "stamp"
        continue
    if stripped == "angular_velocity:":
        section = "gyro"
        continue
    if stripped == "linear_acceleration:":
        section = "accel"
        continue
    if stripped.startswith("frame_id:"):
        current["frame_id"] = stripped.split(":", 1)[1].strip().strip("'")
        continue
    match = re.match(r"^\s*(sec|nanosec|x|y|z):\s*([-+0-9.eE]+)\s*$", line)
    if not match:
        continue
    key, value = match.groups()
    if section == "stamp" and key in ("sec", "nanosec"):
        current[key] = float(value)
    elif section == "gyro" and key in ("x", "y", "z"):
        current["g" + key] = float(value)
    elif section == "accel" and key in ("x", "y", "z"):
        current["a" + key] = float(value)
finish()

if len(samples) < 3:
    print(sys.argv[3] + "_ACCEL_INVALID")
    print("IMU_SANITY_REASON=insufficient_samples")
    raise SystemExit(1)

stamps = [s["sec"] * 1e9 + s["nanosec"] for s in samples]
norms = [math.sqrt(s["ax"] ** 2 + s["ay"] ** 2 + s["az"] ** 2) for s in samples]
numbers = [value for sample in samples for key, value in sample.items()
           if key != "frame_id"]
if (not all(math.isfinite(value) for value in numbers) or
        any(later <= earlier for earlier, later in zip(stamps, stamps[1:])) or
        not all(math.isfinite(value) for value in norms) or
        sum(value >= 0.5 for value in norms) * 2 < len(norms) or
        max(norms) < 0.5 or
        (sys.argv[2] and any(sample.get("frame_id") != sys.argv[2] for sample in samples))):
    print(sys.argv[3] + "_ACCEL_INVALID")
    print("IMU_SANITY_REASON=nonfinite_or_nonmonotonic_or_near_zero_accel")
    raise SystemExit(1)

print("IMU_RAW_SAMPLES=%d" % len(samples))
print("IMU_ACCEL_NORM_MIN=%.6f" % min(norms))
print("IMU_ACCEL_NORM_MEAN=%.6f" % (sum(norms) / len(norms)))
print("IMU_ACCEL_NORM_MAX=%.6f" % max(norms))
print("IMU_TIMESTAMP_MONOTONIC=PASS")
print("IMU_GYRO=PASS")
print("IMU_ACCEL=PASS")
if sys.argv[2]:
    print("IMU_FRAME=%s" % sys.argv[2])
print("IMU_SANITY=PASS")
PY
  )"; then
    printf '%s\n' "$result"
    rm -f "$tmp"
    say "${label}=FAIL"
    say "${label}_ACCEL_INVALID"
    say "POINT_LIO_START_BLOCKED"
    die "ROOT_CAUSE=${label}_ACCEL_INVALID ($topic)"
  fi
  printf '%s\n' "$result"
  rm -f "$tmp"
}

check_transformed_raw_imu() {
  check_imu_stream /sea_nav/lio/transformed_raw_imu TRANSFORMED_RAW_IMU body
  say "TRANSFORMED_RAW_IMU=PASS"
}

transformed_raw_imu_failure_context() {
  say "TRANSFORMED_RAW_IMU=FAIL"
  say "POINT_LIO_STATUS_BEGIN"
  [[ -f "$POINT_STATUS" ]] && cat "$POINT_STATUS" || say "POINT_STATUS_MISSING=$POINT_STATUS"
  say "POINT_LIO_STATUS_END"
  if [[ -f "$POINT_LOG" ]]; then
    say "POINT_LIO_LOG_TAIL_BEGIN=$POINT_LOG"
    tail -n 60 "$POINT_LOG"
    say "POINT_LIO_LOG_TAIL_END"
  else
    say "POINT_LIO_LOG_MISSING=$POINT_LOG"
  fi
  say "POINT_LIO_PROCESS_SNAPSHOT_BEGIN"
  point_lio_pids | while read -r pid; do
    [[ -n "$pid" ]] || continue
    ps -o pid=,stat=,args= -p "$pid" 2>/dev/null || true
  done
  say "POINT_LIO_PROCESS_SNAPSHOT_END"
}

wait_for_transformed_raw_imu() {
  local wait_seconds=10 n pubs state
  for n in $(seq 1 "$wait_seconds"); do
    state="$(status_value "$POINT_STATUS" STATE 2>/dev/null || true)"
    if [[ "$state" == EXITED ]]; then
      transformed_raw_imu_failure_context
      die "ROOT_CAUSE=TRANSFORMED_RAW_IMU_PROCESS_EXITED"
    fi
    pubs="$(publisher_count /sea_nav/lio/transformed_raw_imu)"
    if [[ "$pubs" =~ ^[1-9][0-9]*$ ]]; then
      say "TRANSFORMED_RAW_IMU_PUBLISHER=PASS"
      check_transformed_raw_imu
      return 0
    fi
    say "WAITING_TRANSFORMED_RAW_IMU attempt=${n}/${wait_seconds}"
    sleep 1
  done
  transformed_raw_imu_failure_context
  die "ROOT_CAUSE=TRANSFORMED_RAW_IMU_TIMEOUT"
}

read_latest_odom() {
  local attempt=0 max_attempts=5 result pubs last_error=""
  while (( attempt < max_attempts )); do
    attempt=$((attempt + 1))
    pubs="$(publisher_count /sea_nav/lio/odom_base)"
    : > /tmp/sea_nav_odom_read_error
    set +e
    result="$($PYTHON - <<'PY' 2>/tmp/sea_nav_odom_read_error
import math
import contextlib
import io
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

latest = None

def callback(message):
    global latest
    values = (
        message.pose.pose.position.x,
        message.pose.pose.position.y,
        message.pose.pose.orientation.x,
        message.pose.pose.orientation.y,
        message.pose.pose.orientation.z,
        message.pose.pose.orientation.w,
    )
    if all(math.isfinite(float(value)) for value in values):
        latest = values

diagnostics = io.StringIO()
try:
    with contextlib.redirect_stdout(diagnostics):
        rclpy.init()
        node = Node("sea_nav_forward_odom_reader")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            )
        node.create_subscription(Odometry, "/sea_nav/lio/odom_base", callback, qos)
        deadline = time.monotonic() + 1.5
        while latest is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if latest is None:
            raise RuntimeError("no valid /sea_nav/lio/odom_base sample")
except Exception as exc:
    print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
    raise SystemExit(2)
finally:
    if 'node' in locals():
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
if diagnostics.getvalue():
    print(diagnostics.getvalue(), file=sys.stderr, end="")
print(" ".join("%.17g" % float(value) for value in latest))
PY
)"
    rc=$?
    set -u
    if (( rc == 0 )) && [[ "$result" =~ ^[-+0-9.eE]+[[:space:]]+[-+0-9.eE]+[[:space:]]+[-+0-9.eE]+[[:space:]]+[-+0-9.eE]+[[:space:]]+[-+0-9.eE]+[[:space:]]+[-+0-9.eE]+$ ]]; then
      printf '%s\n' "$result"
      return 0
    fi
    if (( rc == 0 )); then
      last_error="machine-readable odom stdout contract violated"
      say "FORWARD_ODOM_READ_ATTEMPTS=$attempt" >&2
      say "ODOM_BASE_PUBLISHERS=$pubs" >&2
      say "ODOM_BASE_LAST_ERROR=$last_error" >&2
      die "ROOT_CAUSE=FORWARD_ODOM_PARSE_FAILED"
    fi
    last_error="$(cat /tmp/sea_nav_odom_read_error 2>/dev/null || printf '%s' 'no valid odom sample')"
    sleep 1
  done
  say "FORWARD_ODOM_READ_ATTEMPTS=$attempt" >&2
  say "ODOM_BASE_PUBLISHERS=$pubs" >&2
  say "ODOM_BASE_LAST_ERROR=${last_error:-no valid sample}" >&2
  die "ROOT_CAUSE=FORWARD_ODOM_READ_FAILED"
}

resolve_forward_goal() {
  local values current_x current_y qx qy qz qw computed
  [[ -n "$FORWARD" ]] || return 0
  values="$(read_latest_odom)"
  read -r current_x current_y qx qy qz qw <<<"$values"
  say "FORWARD_ODOM_READ=PASS"
  computed="$(/usr/bin/python3 - "$current_x" "$current_y" "$qx" "$qy" "$qz" "$qw" "$FORWARD" <<'PY'
import math
import sys

x, y, qx, qy, qz, qw, forward = map(float, sys.argv[1:])
yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
print(f"{x:.9f} {y:.9f} {math.degrees(yaw):.9f} {x + forward * math.cos(yaw):.9f} {y + forward * math.sin(yaw):.9f}")
PY
  )" || die "ROOT_CAUSE=FORWARD_ODOM_CALC_FAILED"
  read -r current_x current_y current_yaw_deg GOAL_X GOAL_Y <<<"$computed"
  printf 'GOAL_MODE=FORWARD\nFORWARD=%s\nCURRENT_X=%s\nCURRENT_Y=%s\nCURRENT_YAW_DEG=%s\nGOAL_X=%s\nGOAL_Y=%s\n' \
    "$FORWARD" "$current_x" "$current_y" "$current_yaw_deg" "$GOAL_X" "$GOAL_Y" \
    >"$GOAL_STATUS"
  say "FORWARD=$FORWARD CURRENT_X=$current_x CURRENT_Y=$current_y CURRENT_YAW_DEG=$current_yaw_deg GOAL_X=$GOAL_X GOAL_Y=$GOAL_Y"
}

component_pids() {
  local pattern="$1"
  ps -eo pid=,args= | awk -v pattern="$pattern" '
    index($0, pattern) &&
    $0 !~ /go2_lio_selfcheck.sh/ &&
    $0 !~ /go2_lio_diagnose.py/ &&
    $0 !~ /awk -v pattern/ {print $1}'
}

point_lio_pids() {
  {
    cat "$POINT_PID" 2>/dev/null || true
    status_value "$POINT_STATUS" PID 2>/dev/null || true
    status_value "$POINT_STATUS" LAUNCH_PID 2>/dev/null || true
    status_value "$POINT_STATUS" CHILD_PID 2>/dev/null || true
    launch_group_pids
    component_pids "point_lio_go2.launch.py"
    component_pids "pointlio_mapping"
    component_pids "transform_everything"
  } | awk '/^[0-9]+$/ {print}' | sort -nu
}

launch_group_pids() {
  local pgid
  pgid="$(status_value "$POINT_STATUS" LAUNCH_PGID 2>/dev/null || true)"
  [[ "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 ]] || return 0
  ps -eo pid=,pgid= | awk -v pgid="$pgid" '$2 == pgid {print $1}'
}

point_lio_process_tree() {
  local pgid
  pgid="$(status_value "$POINT_STATUS" LAUNCH_PGID 2>/dev/null || true)"
  if [[ "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 ]]; then
    ps -eo pid=,ppid=,pgid=,sid=,stat=,args= | awk -v pgid="$pgid" '$3 == pgid'
  else
    point_lio_pids | while read -r pid; do
      [[ -n "$pid" ]] || continue
      ps -o pid=,ppid=,pgid=,sid=,stat=,args= -p "$pid" 2>/dev/null || true
    done
  fi
}

point_lio_live_pids() {
  local stat
  point_lio_pids | while read -r pid; do
    [[ -n "$pid" ]] || continue
    stat="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    [[ -n "$stat" && "$stat" != Z* ]] && printf '%s\n' "$pid"
  done | sort -nu
}

point_lio_zombie_pids() {
  local stat
  point_lio_pids | while read -r pid; do
    [[ -n "$pid" ]] || continue
    stat="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
    [[ "$stat" == Z* ]] && printf '%s\n' "$pid"
  done | sort -nu
}

pid_alive() {
  local file="$1" pid
  [[ -f "$file" ]] || return 1
  pid="$(<"$file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

status_value() {
  local file="$1" key="$2" value
  [[ -f "$file" ]] || return 1
  value="$(awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$file")"
  [[ -n "$value" ]] || return 1
  printf '%s\n' "$value"
}

write_reused_status() {
  local file="$1" log="$2"
  printf 'STATE=REUSED\nPID=\nEXIT_CODE=\nLOG_FILE=%s\n' "$log" >"$file"
}

wait_topic() {
  local topic="$1" seconds="$2" n
  for n in $(seq 1 "$seconds"); do
    topic_has_data "$topic" && return 0
    sleep 1
  done
  return 1
}

raw_preflight() {
  say "[START] RAW SENSOR PREFLIGHT"
  if topic_has_data /lowstate; then pass "/lowstate"; else fail "/lowstate"; die "ROOT_CAUSE=RAW_SENSOR_FAILURE"; fi
  if topic_has_data /utlidar/cloud; then pass "/utlidar/cloud"; else fail "/utlidar/cloud"; die "ROOT_CAUSE=RAW_SENSOR_FAILURE"; fi
  if topic_has_data /utlidar/imu; then pass "/utlidar/imu"; else fail "/utlidar/imu"; die "ROOT_CAUSE=RAW_SENSOR_FAILURE"; fi
  check_imu_stream /utlidar/imu RAW_IMU ""
  say "RAW_IMU_SANITY=PASS"
}

check_and_release_mcf() {
  local output rc
  [[ -x "$SDK_PYTHON" ]] || die "ROOT_CAUSE=MCF_STATUS_CHECK_FAILED (missing SDK Python)"
  say "[START] MOTION SWITCHER MCF CHECK"
  set +e
  output="$("$SDK_PYTHON" - "$NET" <<'PY'
import sys

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def mode_text(value):
    return "''" if not value else str(value)


net = sys.argv[1]
try:
    ChannelFactoryInitialize(0, net)
    client = MotionSwitcherClient()
    client.SetTimeout(5.0)
    client.Init()

    code, data = client.CheckMode()
    print(f"MCF_INITIAL_CHECK_CODE={code}")
    if code != 0:
        print("ROOT_CAUSE=MCF_STATUS_CHECK_FAILED")
        raise SystemExit(2)

    initial = str((data or {}).get("name", "") or "")
    print(f"MCF_INITIAL_MODE={mode_text(initial)}")
    if initial == "":
        print("MCF_RELEASE=NOT_NEEDED")
        print("MCF_RELEASE_RESULT=PASS")
        print("MCF_FINAL_MODE=''" )
        raise SystemExit(0)

    if initial != "mcf":
        print("MCF_RELEASE=NOT_REQUESTED")
        print("MCF_RELEASE_RESULT=FAIL")
        print(f"MCF_FINAL_MODE={mode_text(initial)}")
        print("ROOT_CAUSE=MCF_RELEASE_FAILED")
        raise SystemExit(3)

    print("MCF_RELEASE=REQUESTED")
    release_code, _ = client.ReleaseMode()
    print(f"MCF_RELEASE_CODE={release_code}")
    if release_code != 0:
        print("MCF_RELEASE_RESULT=FAIL")
        print("MCF_FINAL_MODE=mcf")
        print("ROOT_CAUSE=MCF_RELEASE_FAILED")
        raise SystemExit(4)

    final_code, final_data = client.CheckMode()
    print(f"MCF_FINAL_CHECK_CODE={final_code}")
    final = str((final_data or {}).get("name", "") or "")
    print("MCF_RELEASE_RESULT=PASS" if final_code == 0 and final == "" else "MCF_RELEASE_RESULT=FAIL")
    print(f"MCF_FINAL_MODE={mode_text(final)}")
    if final_code != 0:
        print("ROOT_CAUSE=MCF_STATUS_CHECK_FAILED")
        raise SystemExit(5)
    if final != "":
        print("ROOT_CAUSE=MCF_RELEASE_FAILED")
        raise SystemExit(6)
except SystemExit:
    raise
except Exception as exc:
    print(f"MCF_EXCEPTION={type(exc).__name__}: {exc}")
    print("ROOT_CAUSE=MCF_STATUS_CHECK_FAILED")
    raise SystemExit(7)
PY
  )"
  rc=$?
  set -u
  printf '%s\n' "$output" | tee "$MCF_STATUS_FILE"
  if (( rc != 0 )); then
    if grep -q '^ROOT_CAUSE=MCF_STATUS_CHECK_FAILED$' "$MCF_STATUS_FILE"; then
      die "ROOT_CAUSE=MCF_STATUS_CHECK_FAILED"
    fi
    die "ROOT_CAUSE=MCF_RELEASE_FAILED"
  fi
  grep -q '^MCF_FINAL_MODE=' "$MCF_STATUS_FILE" || die "ROOT_CAUSE=MCF_STATUS_CHECK_FAILED"
  grep -q "^MCF_FINAL_MODE=''$" "$MCF_STATUS_FILE" || die "ROOT_CAUSE=MCF_RELEASE_FAILED"
  pass "MCF_FINAL_MODE=''"
}

raw_after_mcf_release() {
  say "[START] RAW SENSOR CHECK AFTER MCF RELEASE"
  if topic_has_data /lowstate && topic_has_data /utlidar/cloud && topic_has_data /utlidar/imu; then
    pass "RAW_AFTER_MCF_RELEASE=PASS"
  else
    die "ROOT_CAUSE=RAW_SENSOR_FAILURE_AFTER_MCF_RELEASE"
  fi
}

start_terminal() { gnome-terminal --title="$1" -- bash -lc "$2"; }

point_body="
cd $(quote "$LIO_WS")
set +u
source /opt/ros/humble/setup.bash
source $(quote "$UNITREE_SETUP")
source $(quote "$LIO_SETUP")
set -u
export RMW_IMPLEMENTATION=$(quote "$RMW")
export ROS_DOMAIN_ID=$(quote "$ROS_DOMAIN")
export ROS_LOCALHOST_ONLY=$(quote "$ROS_LOCALHOST")
export CYCLONEDDS_URI=$(quote "$CYCLONE")
printf 'STATE=STARTING\\nOWNED=YES\\nPID=\\nEXIT_CODE=\\nLOG_FILE=%s\\n' $(quote "$POINT_LOG") > $(quote "$POINT_STATUS")
echo '[POINT-LIO] ros2 launch sea_nav_lio_bringup point_lio_go2.launch.py'
set +e
setsid ros2 launch sea_nav_lio_bringup point_lio_go2.launch.py > >(tee $(quote "$POINT_LOG")) 2>&1 &
child_pid=\$!
launch_child_pid=\$(pgrep -P \$child_pid | awk 'NR == 1 {print \$1}')
launch_pgid=\$(ps -o pgid= -p \$child_pid | tr -d ' ')
printf 'STATE=RUNNING\\nPID=%s\\nLAUNCH_PID=%s\\nCHILD_PID=%s\\nLAUNCH_PGID=%s\\nEXIT_CODE=\\nLOG_FILE=%s\\n' \$child_pid \$child_pid \${launch_child_pid:-} \${launch_pgid:-} $(quote "$POINT_LOG") > $(quote "$POINT_STATUS")
echo \$child_pid > $(quote "$POINT_PID")
wait \$child_pid
rc=\$?
printf 'STATE=EXITED\\nPID=%s\\nLAUNCH_PID=%s\\nCHILD_PID=%s\\nLAUNCH_PGID=%s\\nEXIT_CODE=%s\\nLOG_FILE=%s\\n' \$child_pid \$child_pid \${launch_child_pid:-} \${launch_pgid:-} \$rc $(quote "$POINT_LOG") > $(quote "$POINT_STATUS")
echo '[POINT-LIO] PROCESS_EXITED'
echo '[POINT-LIO] EXIT_CODE='\$rc
echo '[POINT-LIO] LOG_FILE=$(quote "$POINT_LOG")'
tail -n 50 $(quote "$POINT_LOG")
$(printf '%s' "$TERMINAL_EPILOGUE")
"

adapter_body="
cd $(quote "$LIO_WS")
set +u
source /opt/ros/humble/setup.bash
source $(quote "$UNITREE_SETUP")
source $(quote "$LIO_SETUP")
set -u
export RMW_IMPLEMENTATION=$(quote "$RMW")
export ROS_DOMAIN_ID=$(quote "$ROS_DOMAIN")
export ROS_LOCALHOST_ONLY=$(quote "$ROS_LOCALHOST")
export CYCLONEDDS_URI=$(quote "$CYCLONE")
printf 'STATE=STARTING\\nPID=\\nEXIT_CODE=\\nLOG_FILE=%s\\n' $(quote "$ADAPTER_LOG") > $(quote "$ADAPTER_STATUS")
echo '[ADAPTER] ros2 run sea_nav_lio_bringup odom_se2_adapter --calibration-yaml $(quote "$CALIB")'
set +e
ros2 run sea_nav_lio_bringup odom_se2_adapter \\
  --calibration-yaml $(quote "$CALIB") > >(tee $(quote "$ADAPTER_LOG")) 2>&1 &
child_pid=\$!
printf 'STATE=RUNNING\\nPID=%s\\nEXIT_CODE=\\nLOG_FILE=%s\\n' \$child_pid $(quote "$ADAPTER_LOG") > $(quote "$ADAPTER_STATUS")
echo \$child_pid > $(quote "$ADAPTER_PID")
wait \$child_pid
rc=\$?
printf 'STATE=EXITED\\nPID=%s\\nEXIT_CODE=%s\\nLOG_FILE=%s\\n' \$child_pid \$rc $(quote "$ADAPTER_LOG") > $(quote "$ADAPTER_STATUS")
echo '[ADAPTER] PROCESS_EXITED'
echo '[ADAPTER] EXIT_CODE='\$rc
echo '[ADAPTER] LOG_FILE=$(quote "$ADAPTER_LOG")'
tail -n 50 $(quote "$ADAPTER_LOG")
$(printf '%s' "$TERMINAL_EPILOGUE")
"

bridge_body="
cd $(quote "$REPO")
set +u
source /opt/ros/humble/setup.bash
source $(quote "$UNITREE_SETUP")
set -u
export RMW_IMPLEMENTATION=$(quote "$RMW")
export ROS_DOMAIN_ID=$(quote "$ROS_DOMAIN")
export ROS_LOCALHOST_ONLY=$(quote "$ROS_LOCALHOST")
export CYCLONEDDS_URI=$(quote "$CYCLONE")
printf 'STATE=STARTING\\nPID=\\nEXIT_CODE=\\nLOG_FILE=%s\\n' $(quote "$BRIDGE_LOG") > $(quote "$BRIDGE_STATUS")
echo '[BRIDGE] READ-ONLY; lidar=/utlidar/cloud odom=/sea_nav/lio/odom_base'
set +e
$(quote "$PYTHON") -m deploy.go2_onboard.sensor_bridge \\
  --duration 0 --control-hz 50 --summary-interval 2 \\
  --lowstate-max-age 0.10 --odom-max-age 0.10 --lidar-max-age 0.20 \\
  --lowstate-topic /lowstate --lidar-topic /utlidar/cloud \\
  --odom-topic /sea_nav/lio/odom_base --goal-topic '' __BRIDGE_GOAL_ARGS__ \\
  --socket /tmp/sea_nav_shadow.sock --log $(quote "$BRIDGE_LOG") \\
  > >(tee $(quote "$BRIDGE_LOG")) 2>&1 &
child_pid=\$!
printf 'STATE=RUNNING\\nPID=%s\\nEXIT_CODE=\\nLOG_FILE=%s\\n' \$child_pid $(quote "$BRIDGE_LOG") > $(quote "$BRIDGE_STATUS")
echo \$child_pid > $(quote "$BRIDGE_PID")
wait \$child_pid
rc=\$?
printf 'STATE=EXITED\\nPID=%s\\nEXIT_CODE=%s\\nLOG_FILE=%s\\n' \$child_pid \$rc $(quote "$BRIDGE_LOG") > $(quote "$BRIDGE_STATUS")
echo '[BRIDGE] PROCESS_EXITED'
echo '[BRIDGE] EXIT_CODE='\$rc
echo '[BRIDGE] LOG_FILE=$(quote "$BRIDGE_LOG")'
tail -n 50 $(quote "$BRIDGE_LOG")
$(printf '%s' "$TERMINAL_EPILOGUE")
"

diagnostic_body="
cd $(quote "$REPO")
set +u
source /opt/ros/humble/setup.bash
source $(quote "$UNITREE_SETUP")
set -u
export RMW_IMPLEMENTATION=$(quote "$RMW")
export ROS_DOMAIN_ID=$(quote "$ROS_DOMAIN")
export ROS_LOCALHOST_ONLY=$(quote "$ROS_LOCALHOST")
export CYCLONEDDS_URI=$(quote "$CYCLONE")
printf 'STATE=STARTING\\nPID=\\nEXIT_CODE=\\nLOG_FILE=%s\\n' $(quote "$DIAG_LOG") > $(quote "$DIAG_STATUS")
echo '[DIAGNOSTIC] read-only monitor; no publisher is created'
set +e
$(quote "$PYTHON") $(quote "$REPO/tools/go2_lio_diagnose.py") \\
  --report-interval 2 --status-file $(quote "$STATUS_FILE") \\
  --point-pid $(quote "$POINT_PID") --adapter-pid $(quote "$ADAPTER_PID") \\
  --bridge-pid $(quote "$BRIDGE_PID") > >(tee $(quote "$DIAG_LOG")) 2>&1 &
child_pid=\$!
printf 'STATE=RUNNING\\nPID=%s\\nEXIT_CODE=\\nLOG_FILE=%s\\n' \$child_pid $(quote "$DIAG_LOG") > $(quote "$DIAG_STATUS")
echo \$child_pid > $(quote "$DIAG_PID")
wait \$child_pid
rc=\$?
printf 'STATE=EXITED\\nPID=%s\\nEXIT_CODE=%s\\nLOG_FILE=%s\\n' \$child_pid \$rc $(quote "$DIAG_LOG") > $(quote "$DIAG_STATUS")
echo '[DIAGNOSTIC] PROCESS_EXITED'
echo '[DIAGNOSTIC] EXIT_CODE='\$rc
echo '[DIAGNOSTIC] LOG_FILE=$(quote "$DIAG_LOG")'
tail -n 50 $(quote "$DIAG_LOG")
$(printf '%s' "$TERMINAL_EPILOGUE")
"

start_or_reuse_point() {
  local pubs existing
  pubs="$(publisher_count /sea_nav/lio/odom)"; [[ "$pubs" =~ ^[0-9]+$ ]] || pubs=0
  existing="$(component_pids pointlio_mapping; component_pids transform_everything; component_pids point_lio_go2.launch.py; ros2 node list 2>/dev/null | grep -E '^/sea_nav_(point_lio|sensor_transform)$' || true)"
  if (( pubs > 1 )); then die "ROOT_CAUSE=DUPLICATE_PUBLISHER (Point-LIO)"; fi
  if (( pubs == 1 )) && topic_has_data /sea_nav/lio/odom; then
    POINT_LIO_STARTED=0
    write_reused_status "$POINT_STATUS" "$POINT_LOG"
    say "EXISTING_PROCESS=HEALTHY Point-LIO; reusing"
    wait_for_transformed_raw_imu
    return 0
  fi
  [[ -z "$existing" ]] || die "EXISTING_PROCESS=UNHEALTHY Point-LIO; refusing duplicate launch"
  say "[START] POINT_LIO"
  POINT_LIO_STARTED=1
  printf 'STATE=STARTING\nOWNED=YES\nPID=\nLAUNCH_PID=\nCHILD_PID=\nLAUNCH_PGID=\nEXIT_CODE=\nLOG_FILE=%s\n' \
    "$POINT_LOG" >"$POINT_STATUS"
  start_terminal "Go2 Point-LIO (READ ONLY)" "$point_body"
  wait_for_transformed_raw_imu
  wait_topic /sea_nav/lio/odom 30 || die "ROOT_CAUSE=POINT_LIO_NO_OUTPUT"
  [[ "$(publisher_count /sea_nav/lio/odom)" == 1 ]] || die "ROOT_CAUSE=DUPLICATE_PUBLISHER (Point-LIO)"
  pass "process /sea_nav/lio/odom real data"
}

start_or_reuse_adapter() {
  local pubs existing
  pubs="$(publisher_count /sea_nav/lio/odom_base)"; [[ "$pubs" =~ ^[0-9]+$ ]] || pubs=0
  existing="$(component_pids odom_se2_adapter; ros2 node list 2>/dev/null | grep -qx /sea_nav_lio_odom_se2_adapter || true)"
  if (( pubs > 1 )); then die "ROOT_CAUSE=DUPLICATE_PUBLISHER (adapter)"; fi
  if (( pubs == 1 )) && topic_has_data /sea_nav/lio/odom_base; then
    write_reused_status "$ADAPTER_STATUS" "$ADAPTER_LOG"
    say "EXISTING_PROCESS=HEALTHY adapter; reusing"
    return 0
  fi
  [[ -z "$existing" ]] || die "ROOT_CAUSE=ADAPTER_FAILURE (unhealthy existing adapter)"
  say "[START] ADAPTER"
  start_terminal "Go2 Odom Adapter (READ ONLY)" "$adapter_body"
  wait_topic /sea_nav/lio/odom_base 20 || die "ROOT_CAUSE=ADAPTER_NO_OUTPUT"
  [[ "$(publisher_count /sea_nav/lio/odom_base)" == 1 ]] || die "ROOT_CAUSE=DUPLICATE_PUBLISHER (adapter)"
  timeout 4s ros2 topic echo /sea_nav/lio/odom_base --once 2>/dev/null | grep -q 'frame_id: odom' || die "ROOT_CAUSE=ADAPTER_BAD_FRAME"
  timeout 4s ros2 topic echo /sea_nav/lio/odom_base --once 2>/dev/null | grep -q 'child_frame_id: base_link' || die "ROOT_CAUSE=ADAPTER_BAD_FRAME"
  pass "process /sea_nav/lio/odom_base real data frame odom -> base_link"
}

start_or_reuse_bridge() {
  local existing node_info body
  existing="$(component_pids deploy.go2_onboard.sensor_bridge)"
  node_info="$(ros2 node info /sea_nav_go2_shadow_runtime 2>/dev/null || true)"
  if [[ -n "$existing" ]]; then
    [[ -z "$BRIDGE_GOAL_ARGS" ]] || die "ROOT_CAUSE=SENSOR_BRIDGE_GOAL_UNKNOWN (existing bridge cannot be reused with a new goal)"
    if ros2 node list 2>/dev/null | grep -qx /sea_nav_go2_shadow_runtime && \
       grep -Eq '^[[:space:]]*/sea_nav/lio/odom_base([[:space:]]|:)' <<<"$node_info" && \
       ! grep -Eq '^[[:space:]]*/utlidar/cloud_base([[:space:]]|:)' <<<"$node_info" && \
       grep -Eq '^[[:space:]]*/utlidar/cloud([[:space:]]|:)' <<<"$node_info"; then
      write_reused_status "$BRIDGE_STATUS" "$BRIDGE_LOG"
      say "EXISTING_PROCESS=HEALTHY sensor_bridge; reusing"
      return 0
    fi
    die "ROOT_CAUSE=SENSOR_BRIDGE_NOT_RUNNING (unhealthy existing bridge)"
  fi
  if [[ -n "$FORWARD" ]]; then
    resolve_forward_goal
    set_bridge_goal
  fi
  say "[START] SENSOR_BRIDGE"
  body="${bridge_body//__BRIDGE_GOAL_ARGS__/$BRIDGE_GOAL_ARGS}"
  start_terminal "Go2 Sensor Bridge (READ ONLY)" "$body"
  for _ in $(seq 1 15); do
    if ros2 node list 2>/dev/null | grep -qx /sea_nav_go2_shadow_runtime; then break; fi
    sleep 1
  done
  ros2 node list 2>/dev/null | grep -qx /sea_nav_go2_shadow_runtime || die "ROOT_CAUSE=SENSOR_BRIDGE_NOT_RUNNING"
  node_info="$(ros2 node info /sea_nav_go2_shadow_runtime 2>/dev/null || true)"
  grep -Eq '^[[:space:]]*/sea_nav/lio/odom_base([[:space:]]|:)' <<<"$node_info" || die "ROOT_CAUSE=SENSOR_BRIDGE_ODOM_NOT_SUBSCRIBED"
  if grep -Eq '^[[:space:]]*/utlidar/cloud_base([[:space:]]|:)' <<<"$node_info" || \
     ! grep -Eq '^[[:space:]]*/utlidar/cloud([[:space:]]|:)' <<<"$node_info"; then
    die "ROOT_CAUSE=SENSOR_BRIDGE_WRONG_LIDAR_TOPIC"
  fi
  pass "node/subscriptions: odom=/sea_nav/lio/odom_base lidar=/utlidar/cloud"
}

start_diagnostic() {
  say "[START] DIAGNOSTIC"
  start_terminal "Go2 LIO Diagnostic Monitor (READ ONLY)" "$diagnostic_body"
  for _ in $(seq 1 10); do
    [[ "$(status_value "$DIAG_STATUS" STATE 2>/dev/null || true)" == RUNNING ]] && return 0
    sleep 1
  done
  die "ROOT_CAUSE=DIAGNOSTIC_PROCESS_EXITED LOG_FILE=$DIAG_LOG"
}

last_log_lines() {
  local file="$1"
  [[ -f "$file" ]] && tail -n 8 "$file"
}

bridge_metric() {
  local key="$1"
  if [[ -f "$BRIDGE_LOG" ]] && grep -Eq "${key}=[1-9]|\"${key}\"[[:space:]]*:[[:space:]]*[1-9]" "$BRIDGE_LOG"; then
    echo PASS
  else
    echo UNKNOWN
  fi
}

bridge_lidar_subscription() {
  local node_info
  node_info="$(ros2 node info /sea_nav_go2_shadow_runtime 2>/dev/null || true)"
  if grep -Eq '^[[:space:]]*/utlidar/cloud_base([[:space:]]|:)' <<<"$node_info"; then
    echo FAIL
  elif grep -Eq '^[[:space:]]*/utlidar/cloud([[:space:]]|:)' <<<"$node_info"; then
    echo PASS
  else
    echo FAIL
  fi
}

supervisor_root_cause() {
  local state lio_pubs base_pubs
  state="$(status_value "$DIAG_STATUS" STATE 2>/dev/null || true)"
  if [[ "$state" == EXITED ]]; then echo "DIAGNOSTIC_PROCESS_EXITED"; return; fi
  state="$(status_value "$POINT_STATUS" STATE 2>/dev/null || true)"
  if [[ "$state" == EXITED ]]; then echo "POINT_LIO_PROCESS_EXITED"; return; fi
  state="$(status_value "$ADAPTER_STATUS" STATE 2>/dev/null || true)"
  if [[ "$state" == EXITED ]]; then echo "ADAPTER_PROCESS_EXITED"; return; fi
  state="$(status_value "$BRIDGE_STATUS" STATE 2>/dev/null || true)"
  if [[ "$state" == EXITED ]]; then echo "SENSOR_BRIDGE_PROCESS_EXITED"; return; fi
  lio_pubs="$(publisher_count /sea_nav/lio/odom)"; [[ "$lio_pubs" =~ ^[0-9]+$ ]] || lio_pubs=0
  base_pubs="$(publisher_count /sea_nav/lio/odom_base)"; [[ "$base_pubs" =~ ^[0-9]+$ ]] || base_pubs=0
  if (( lio_pubs > 1 || base_pubs > 1 )); then
    echo "DUPLICATE_PUBLISHER"; return
  fi
  status_value "$STATUS_FILE" DIAGNOSIS 2>/dev/null || echo "STARTING"
}

show_supervisor() {
  local root="" state diag_state
  printf '\n================ GO2 LIO SUPERVISOR ================\n'
  if [[ -f "$STATUS_FILE" ]]; then
    grep -E '^(RAW_LOWSTATE|RAW_CLOUD|RAW_IMU|POINT_LIO_PROCESS|POINT_LIO_PUBS|POINT_LIO_DATA|POINT_LIO_RATE|ADAPTER_PROCESS|ODOM_BASE_PUBS|ODOM_BASE_DATA|ODOM_BASE_RATE|ODOM_BASE_FRAME|SENSOR_BRIDGE|BRIDGE_ODOM_SUB|DIAGNOSIS|DRIFT_CHECK)' "$STATUS_FILE" || true
  else
    echo 'DIAGNOSTIC        = STARTING'
  fi
  [[ -f "$MCF_STATUS_FILE" ]] && grep -E '^(MCF_INITIAL_MODE|MCF_RELEASE=|MCF_RELEASE_RESULT|MCF_FINAL_MODE|ROOT_CAUSE=)' "$MCF_STATUS_FILE" || true
  diag_state="$(status_value "$DIAG_STATUS" STATE 2>/dev/null || true)"
  [[ -n "$diag_state" ]] && echo "DIAGNOSTIC        = $diag_state"
  echo "BRIDGE_LIDAR_SUB  = $(bridge_lidar_subscription)"
  echo "BRIDGE_LOWSTATE_DATA = $(bridge_metric lowstate_total)"
  echo "BRIDGE_ODOM_DATA     = $(bridge_metric odom_total)"
  root="$(supervisor_root_cause)"
  echo "ROOT_CAUSE        = $root"
  echo "LOG_DIR           = $LOGROOT"
  echo '====================================================='
  if [[ "$root" != PASS && "$root" != STARTING && "$root" != "$LAST_ROOT_CAUSE" ]]; then
    fail "${root}"
    case "$root" in
      DIAGNOSTIC_PROCESS_EXITED) echo "EXIT_CODE=$(status_value "$DIAG_STATUS" EXIT_CODE 2>/dev/null || true)"; echo "LOG_FILE=$DIAG_LOG"; last_log_lines "$DIAG_LOG";;
      POINT_LIO_PROCESS_EXITED) echo "EXIT_CODE=$(status_value "$POINT_STATUS" EXIT_CODE 2>/dev/null || true)"; echo "LOG_FILE=$POINT_LOG"; last_log_lines "$POINT_LOG";;
      ADAPTER_PROCESS_EXITED) echo "EXIT_CODE=$(status_value "$ADAPTER_STATUS" EXIT_CODE 2>/dev/null || true)"; echo "LOG_FILE=$ADAPTER_LOG"; last_log_lines "$ADAPTER_LOG";;
      SENSOR_BRIDGE_PROCESS_EXITED) echo "EXIT_CODE=$(status_value "$BRIDGE_STATUS" EXIT_CODE 2>/dev/null || true)"; echo "LOG_FILE=$BRIDGE_LOG"; last_log_lines "$BRIDGE_LOG";;
    esac
  fi
  LAST_ROOT_CAUSE="$root"
}

LAST_ROOT_CAUSE=""
CLEANUP_STARTED=0

wait_pid_exit() {
  local pid="$1" loops="${2:-50}"
  for _ in $(seq 1 "$loops"); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  return 1
}

stop_owned_component() {
  local label="$1" status_file="$2" pid_file="$3" log_file="$4"
  local state pid
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
  say "[STOP] $label pid=$pid"
  kill -INT "$pid" 2>/dev/null || true
  if wait_pid_exit "$pid" 50; then
    say "[PASS] ${label}_STOPPED"
    return 0
  fi
  say "[WARN] $label pid=$pid did not exit after SIGINT; log=${log_file}"
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
  fail "$label pid=$pid could not be stopped; log=${log_file}"
  return 1
}

stop_owned_pid() {
  local label="$1" pid="$2" log_file="$3"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  say "[STOP] $label pid=$pid"
  kill -INT "$pid" 2>/dev/null || true
  if wait_pid_exit "$pid" 50; then
    say "[PASS] ${label}_STOPPED"
    return 0
  fi
  say "[WARN] $label pid=$pid did not exit after SIGINT; log=${log_file}"
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
  fail "$label pid=$pid could not be stopped; log=${log_file}"
  return 1
}

process_group_empty() {
  local pgid="$1"
  ! ps -eo pid=,pgid=,stat= | awk -v pgid="$pgid" '$2 == pgid && $3 !~ /^Z/ {found=1} END {exit found ? 0 : 1}'
}

process_group_report() {
  local label="$1" pgid="$2" rows pid ppid actual_pgid stat cmd live=0 zombie=0
  rows="$(ps -eo pid=,ppid=,pgid=,stat=,args= | awk -v pgid="$pgid" '$3 == pgid')"
  if [[ -z "$rows" ]]; then
    say "${label}=EMPTY"
    return 0
  fi
  while read -r pid ppid actual_pgid stat cmd; do
    [[ -n "$pid" ]] || continue
    if [[ "$stat" == Z* ]]; then
      zombie=$((zombie + 1))
      say "${label}=ZOMBIE_WAITING_REAP PID=$pid PPID=$ppid PGID=$actual_pgid STAT=$stat CMD=$cmd"
    else
      live=$((live + 1))
      say "${label}=LIVE_PROCESS_REMAINS PID=$pid PPID=$ppid PGID=$actual_pgid STAT=$stat CMD=$cmd"
    fi
  done <<<"$rows"
  say "${label}_SUMMARY=live:$live zombie:$zombie"
}

stop_owned_process_group() {
  local label="$1" pgid="$2" log_file="$3"
  [[ "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 ]] || return 1
  say "[STOP] $label process_group=$pgid"
  kill -INT -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    if process_group_empty "$pgid"; then
      process_group_report "POINT_LIO_GROUP_AFTER_SIGINT" "$pgid"
      say "[PASS] ${label}_GROUP_STOPPED"
      return 0
    fi
    sleep 0.1
  done
  process_group_report "POINT_LIO_GROUP_AFTER_SIGINT" "$pgid"
  say "[WARN] $label process_group=$pgid did not exit after SIGINT; log=$log_file"
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if process_group_empty "$pgid"; then
      process_group_report "POINT_LIO_GROUP_AFTER_SIGTERM" "$pgid"
      say "[PASS] ${label}_GROUP_STOPPED_AFTER_SIGTERM"
      return 0
    fi
    sleep 0.1
  done
  process_group_report "POINT_LIO_GROUP_AFTER_SIGTERM" "$pgid"
  say "[WARN] $label process_group=$pgid still alive; escalating to SIGKILL"
  kill -KILL -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if process_group_empty "$pgid"; then
      process_group_report "POINT_LIO_GROUP_AFTER_SIGKILL" "$pgid"
      say "[PASS] ${label}_GROUP_STOPPED_AFTER_SIGKILL"
      return 0
    fi
    sleep 0.1
  done
  process_group_report "POINT_LIO_GROUP_AFTER_SIGKILL" "$pgid"
  if process_group_empty "$pgid"; then
    say "[PASS] ${label}_GROUP_STOPPED_ZOMBIE_ONLY"
    return 0
  fi
  fail "$label process_group=$pgid could not be stopped; log=$log_file"
  return 1
}

stop_point_lio() {
  local state pids pid pgid
  say "[STOP] POINT_LIO"
  state="$(status_value "$POINT_STATUS" STATE 2>/dev/null || true)"
  if [[ "$state" == REUSED && "$POINT_LIO_STARTED" -eq 0 ]]; then
    say "[PASS] POINT_LIO_NOT_OWNED (REUSED)"
    return 0
  fi
  pgid="$(status_value "$POINT_STATUS" LAUNCH_PGID 2>/dev/null || true)"
  if (( POINT_LIO_STARTED )) && [[ "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 ]]; then
    say "[PROCESS_TREE] POINT_LIO_BEGIN"
    point_lio_process_tree
    say "[PROCESS_TREE] POINT_LIO_END"
    stop_owned_process_group "POINT_LIO" "$pgid" "$POINT_LOG" || return 1
  fi
  pids="$(point_lio_live_pids)"
  if [[ -z "$pids" ]]; then
    if (( POINT_LIO_STARTED )); then
      say "[PASS] POINT_LIO_STOPPED (no owned process remained)"
    else
      say "[PASS] POINT_LIO_NOT_RUNNING"
    fi
    return 0
  fi
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    stop_owned_pid "POINT_LIO" "$pid" "$POINT_LOG" || return 1
  done <<<"$pids"
  if [[ -n "$(point_lio_live_pids)" ]]; then
    fail "POINT_LIO orphan remains; log=$POINT_LOG"
    return 1
  fi
  if [[ -n "$(point_lio_zombie_pids)" ]]; then
    say "POINT_LIO_ZOMBIE_WAITING_REAP=$(point_lio_zombie_pids | tr '\n' ' ')"
  fi
  say "[PASS] POINT_LIO_STOPPED"
}

cleanup_owned_children() {
  (( CLEANUP_STARTED == 0 )) || return 0
  CLEANUP_STARTED=1
  echo
  say "CLEANUP_START log_dir=$LOGROOT"
  stop_owned_component "DIAGNOSTIC" "$DIAG_STATUS" "$DIAG_PID" "$DIAG_LOG"
  stop_owned_component "SENSOR_BRIDGE" "$BRIDGE_STATUS" "$BRIDGE_PID" "$BRIDGE_LOG"
  stop_owned_component "ADAPTER" "$ADAPTER_STATUS" "$ADAPTER_PID" "$ADAPTER_LOG"
  stop_point_lio
  say "CLEANUP_DONE"
}

on_stop() {
  cleanup_owned_children
  exit 0
}

on_exit() {
  cleanup_owned_children
}

trap on_stop INT TERM
trap on_exit EXIT

raw_preflight
check_and_release_mcf
raw_after_mcf_release
start_or_reuse_point
start_or_reuse_adapter
start_or_reuse_bridge
start_diagnostic

say "CENTRAL SUPERVISOR active; press Ctrl+C to stop owned components"
while :; do
  show_supervisor
  sleep 2
done

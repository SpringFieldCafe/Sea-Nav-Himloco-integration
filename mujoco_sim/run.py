import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from deploy.go2_onboard.himloco_observation import HIMLocoObservation
from deploy.go2_onboard.model_loader import infer, load_himloco_policy, load_navigation_policy

from .adapters import MuJoCoStateAdapter
from .core import PolicyRuntimeCore
from .supervisor import TerrainSupervisor
from .visualization import WaypointVisualizer
from .waypoints import WaypointManager


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "go2"
SCENES = {
    "himloco_flat": ASSETS / "scene.xml",
    "stairs_up": ASSETS / "stair.xml",
    "stairs_down": ASSETS / "stairs_down.xml",
    "rough": ASSETS / "hfield.xml",
    "flat_obstacle": ASSETS / "course_flat_obstacle.xml",
    "mixed_course": ASSETS / "course_mixed.xml",
    "winding_course": ASSETS / "course_winding.xml",
    "winding_flat_turn": ASSETS / "course_winding_flat.xml",
}

# Match the original HIMLoco Go2 MuJoCo deployment baseline.  The official
# deploy_mujoco_go2.py applies one symmetric +/-33.5 Nm limit to all joints.
HIMLOCO_TORQUE_LIMITS = np.full(12, 33.5, dtype=np.float64)
HIMLOCO_ACTION_CLIP = 100.0
HIMLOCO_HIP_INDICES = np.asarray([0, 3, 6, 9], dtype=np.int64)
HIMLOCO_HIP_REDUCTION = 1.0


def _himloco_pd_torque(target, q, dq):
    torque = 20.0 * (target - q) - 0.5 * dq
    return np.clip(torque, -HIMLOCO_TORQUE_LIMITS, HIMLOCO_TORQUE_LIMITS)


def _contact_diagnostics(model, data):
    """Summarize foot contact forces and tangential slip without changing dynamics."""
    import mujoco

    feet = {name: model.body(f"{name}_calf").id
            for name in ("FL", "FR", "RL", "RR")}
    result = {name: {"normal_force": 0.0, "tangential_force": 0.0,
                     "tangential_slip": 0.0, "contacts": 0}
              for name in feet}
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body_ids = (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2])
        foot_names = [name for name, body_id in feet.items() if body_id in body_ids]
        if not foot_names:
            continue
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_id, force)
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        # Contact force is expressed in the contact frame: x/y tangent, z normal.
        normal_force = max(0.0, float(force[2]))
        tangential_force = float(np.linalg.norm(force[:2]))
        for name in foot_names:
            body_id = feet[name]
            spatial = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                     body_id, spatial, 0)
            angular_world = frame.T @ np.zeros(3)  # initialized for clarity below
            body_rot = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            angular_world = body_rot @ spatial[:3]
            linear_world = body_rot @ spatial[3:]
            contact_velocity = linear_world + np.cross(
                angular_world, np.asarray(contact.pos) - np.asarray(data.xpos[body_id]))
            tangent_velocity = contact_velocity - frame[2] * float(np.dot(frame[2], contact_velocity))
            result[name]["normal_force"] += normal_force
            result[name]["tangential_force"] += tangential_force
            result[name]["tangential_slip"] = max(
                result[name]["tangential_slip"], float(np.linalg.norm(tangent_velocity)))
            result[name]["contacts"] += 1
    return result


def _set_contact_friction(model, coefficient):
    """Set only floor/foot sliding friction for a one-variable A/B test."""
    coefficient = float(coefficient)
    if coefficient <= 0.0:
        raise ValueError("contact friction must be positive")
    contact_geoms = {model.geom("floor").id}
    for name in ("FL", "FR", "RL", "RR"):
        body_id = model.body(f"{name}_calf").id
        contact_geoms.update(
            i for i in range(model.ngeom)
            if model.geom_bodyid[i] == body_id and model.geom_contype[i] != 0
        )
    for geom_id in contact_geoms:
        model.geom_friction[geom_id, 0] = coefficient


def _contact_geom_ids(model):
    ids = {model.geom("floor").id}
    for name in ("FL", "FR", "RL", "RR"):
        body_id = model.body(f"{name}_calf").id
        ids.update(i for i in range(model.ngeom)
                   if model.geom_bodyid[i] == body_id and model.geom_contype[i] != 0)
    return ids


def _apply_contact_ab(model, args):
    """Apply one optional contact-solver A/B change in memory only."""
    import mujoco
    if args.noslip_iterations is not None:
        model.opt.noslip_iterations = args.noslip_iterations
    if args.friction_cone == "elliptic":
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    if args.impratio is not None:
        model.opt.impratio = args.impratio
    if args.contact_solref is not None:
        for geom_id in _contact_geom_ids(model):
            model.geom_solref[geom_id] = args.contact_solref
    if args.contact_solimp is not None:
        for geom_id in _contact_geom_ids(model):
            model.geom_solimp[geom_id] = args.contact_solimp


def _diagnostic_record(model, data, state, command, action, target):
    body_id = model.body("base").id
    quat = np.asarray(data.xquat[body_id], dtype=np.float64)
    w, x, y, z = quat
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return {
        "actual_vx": float(state.linear_velocity[0]),
        "actual_vy": float(state.linear_velocity[1]),
        "actual_wz": float(state.angular_velocity[2]),
        "actual_linear_velocity": np.asarray(state.linear_velocity).tolist(),
        "actual_angular_velocity": np.asarray(state.angular_velocity).tolist(),
        "gravity": np.asarray(state.gravity).tolist(),
        "roll": float(state.roll), "pitch": float(state.pitch), "yaw": float(yaw),
        "joint_position": np.asarray(state.joint_position).tolist(),
        "joint_velocity": np.asarray(state.joint_velocity).tolist(),
        "policy_action": np.asarray(action).tolist(),
        "target_position": np.asarray(target).tolist(),
        "torque": np.asarray(data.ctrl).tolist(),
        "foot_contacts": _contact_diagnostics(model, data),
    }


def _terrain(scene):
    if scene not in ("mixed_course", "winding_course", "winding_flat_turn"):
        return {"stairs_up": "stairs_up", "stairs_down": "stairs_down",
                "rough": "rough"}.get(scene, "flat")
    return lambda _x: "flat" if scene == "winding_flat_turn" else "rough"


def _reset(model, data, spawn_xy=(0.0, 0.0)):
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qpos[0:2] = np.asarray(spawn_xy, dtype=np.float64)
    # Match HIMLoco's original deploy_mujoco Go2 reset.
    data.qpos[2] = .35
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:19] = [.1, .8, -1.5, -.1, .8, -1.5, .1, 1, -1.5, -.1, 1, -1.5]
    __import__("mujoco").mj_forward(model, data)


def _configure_random_obstacles(model, data, seed, waypoints, count=10):
    """Place extra boxes reproducibly while preserving a center safe corridor."""
    rng = np.random.default_rng(seed)
    max_count = 10
    count = max(0, min(int(count), max_count))
    names = [f"random_box_{i}" for i in range(count)]
    waypoint_xy = np.asarray([w.xy for w in waypoints.waypoints], dtype=np.float32)
    placed = []
    candidates = np.arange(2.2, 19.0, 0.4, dtype=np.float32)
    rng.shuffle(candidates)
    for x in candidates:
        if len(placed) == len(names):
            break
        # Keep the waypoint corridor clear. Obstacles remain offset from the
        # centerline so the policy can choose either side without a dead end.
        y = float(rng.choice((-1.80, -1.60, 1.60, 1.80)))
        size_xy = float(rng.uniform(0.22, 0.32))
        if np.any(np.linalg.norm(waypoint_xy - np.array([x, y]), axis=1)
                  < 0.8 + size_xy + 0.45):
            continue
        # Keep one obstacle per longitudinal slice. Two boxes on opposite
        # sides at nearly the same x create an unintended gate for the local
        # policy even when their Euclidean centers do not overlap.
        if any(abs(float(x) - float(p[0])) < 1.4 for p in placed):
            continue
        placed.append(np.array([x, y], dtype=np.float32))
        geom_id = model.geom(names[len(placed) - 1]).id
        model.geom_pos[geom_id, :2] = [x, y]
        model.geom_size[geom_id, :2] = size_xy
        model.geom_size[geom_id, 2] = float(rng.uniform(0.25, 0.42))
        # The XML placeholders are below the floor. Lift each randomized box
        # above the rough-field height so it is visible and ray-detectable.
        model.geom_pos[geom_id, 2] = model.geom_size[geom_id, 2] + 0.08
    __import__("mujoco").mj_forward(model, data)
    return [{"name": name, "x": float(pos[0]), "y": float(pos[1])}
            for name, pos in zip(names, placed)]


def _run_himloco(model, data, args, adapter, viewer=None, recorder=None):
    policy = load_himloco_policy(args.himloco_policy, torch.device(args.device))
    obs = HIMLocoObservation(torch.device(args.device))
    default = np.array([.1, .8, -1.5, -.1, .8, -1.5, .1, 1, -1.5, -.1, 1, -1.5], dtype=np.float32)
    action = np.zeros(12, dtype=np.float32)
    requested_command = np.asarray([args.vx, args.vy, args.wz], dtype=np.float32)
    # Pre-roll keeps the requested forward speed but suppresses yaw, matching
    # the requested stable straight-walking phase before turning.
    command_target = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    command = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    records = []
    control_steps = max(1, int(round(.02 / model.opt.timestep)))
    for step in range(args.steps):
        q = data.qpos[7:19]
        dq = data.qvel[6:18]
        action_scaled = np.clip(action, -HIMLOCO_ACTION_CLIP, HIMLOCO_ACTION_CLIP) * .25
        action_scaled[HIMLOCO_HIP_INDICES] *= HIMLOCO_HIP_REDUCTION
        data.ctrl[:] = _himloco_pd_torque(default + action_scaled, q, dq)
        __import__("mujoco").mj_step(model, data)
        # Match the official deploy_mujoco loop: infer after a complete
        # decimation window, then apply the new target on the next step.
        if (step + 1) % control_steps == 0:
            elapsed_s = (step + 1) * model.opt.timestep
            if elapsed_s >= args.stable_pre_roll:
                command_target = requested_command
            alpha = float(args.himloco_command_filter_alpha)
            command = alpha * command_target + (1.0 - alpha) * command
            state = adapter.read([args.goal_x, args.goal_y])
            torch_state = {k: torch.from_numpy(v).reshape(1, -1).to(args.device) for k, v in
                           {"command": command, "angular": state.angular_velocity,
                            "gravity": state.gravity, "q": state.joint_position - default,
                            "dq": state.joint_velocity}.items()}
            inp = obs.build(torch_state["command"], torch_state["angular"], torch_state["gravity"], torch_state["q"], torch_state["dq"])
            action = infer(policy, inp, 12)[0].cpu().numpy()
            action = np.clip(action, -HIMLOCO_ACTION_CLIP, HIMLOCO_ACTION_CLIP)
            obs.record_action(torch.from_numpy(action).reshape(1, -1))
            next_target = default + action * .25
            records.append({"step": step, "command_target": command_target.tolist(),
                            "requested_command": requested_command.tolist(),
                            "stable_pre_roll_s": args.stable_pre_roll,
                            "command": command.tolist(), "himloco_action": action.tolist(), "terrain": state.terrain_hint,
                            "position_xy": state.position_xy.tolist(), "roll": state.roll,
                            "pitch": state.pitch, "min_obstacle_distance": state.min_obstacle_distance,
                            "collision": state.collision, "fallen": state.fallen,
                            "diagnostics": _diagnostic_record(
                                model, data, state, command, action, next_target),
                            "observation": inp[0].detach().cpu().numpy().tolist(),
                            "one_step_observation": inp[0, :45].detach().cpu().numpy().tolist(),
                            "command_observation": (command * np.asarray([2.0, 2.0, 0.25])).tolist(),
                            "base_ang_vel_raw": state.angular_velocity.tolist(),
                            "base_ang_vel_observation": (state.angular_velocity * 0.25).tolist(),
                            "projected_gravity_raw": state.gravity.tolist(),
                            "dof_pos_raw": (state.joint_position - default).tolist(),
                            "dof_pos_observation": (state.joint_position - default).tolist(),
                            "dof_vel_raw": state.joint_velocity.tolist(),
                            "dof_vel_observation": (state.joint_velocity * 0.05).tolist()})
        if viewer is not None:
            viewer.sync()
        if recorder is not None and (step + 1) % control_steps == 0:
            recorder.write(model, data)
    return records


def main():
    parser = argparse.ArgumentParser(description="SEA-Nav x HIMLoco MuJoCo terrain runner")
    parser.add_argument("--scene", choices=sorted(SCENES), required=True)
    parser.add_argument("--himloco-policy", required=True)
    parser.add_argument("--navigation-policy", default="")
    parser.add_argument("--goal-x", type=float, default=12.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--vx", type=float, default=.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--himloco-command-filter-alpha", type=float, default=1.0,
                        help="HIMLoco command filter alpha; 0.15 matches continuous-turning training")
    parser.add_argument("--stable-pre-roll", type=float, default=0.0,
                        help="zero-command stabilization time before target command (seconds)")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--record", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--waypoints", default="")
    parser.add_argument("--goal-radius", type=float, default=0.5)
    parser.add_argument("--ray-mode", choices=("grid2ray", "physical_lidar"), default="grid2ray")
    parser.add_argument("--angular-velocity-source", choices=("cvel", "qvel"), default="cvel",
                        help="MuJoCo angular velocity source; qvel matches official HIMLoco deploy")
    parser.add_argument("--contact-friction", type=float, default=None,
                        help="A/B test: set floor and foot sliding friction only")
    parser.add_argument("--noslip-iterations", type=int, default=None,
                        help="A/B test: MuJoCo noslip iterations")
    parser.add_argument("--friction-cone", choices=("pyramidal", "elliptic"), default="pyramidal")
    parser.add_argument("--impratio", type=float, default=None,
                        help="A/B test: elliptic friction cone impedance ratio")
    parser.add_argument("--contact-solref", type=float, nargs=2, default=None,
                        metavar=("TIMEConst", "DAMPING"), help="A/B test contact solref")
    parser.add_argument("--contact-solimp", type=float, nargs=5, default=None,
                        metavar=("D0", "DWidth", "Width", "Midpoint", "Power"),
                        help="A/B test contact solimp")
    parser.add_argument("--draw-goal", action="store_true")
    parser.add_argument("--stop-on-goal", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="scale terrain command limits, capped by HIMLoco bounds")
    parser.add_argument("--command-filter-alpha", type=float, default=0.5,
                        help="new-command weight in (0,1], larger is more responsive")
    parser.add_argument("--no-open-space-assist", action="store_true",
                        help="disable clear-space lateral drift damping")
    parser.add_argument("--random-obstacles", action="store_true",
                        help="place additional seed-reproducible boxes in mixed_course")
    parser.add_argument("--random-obstacle-count", type=int, default=10,
                        help="number of extra boxes, 0-10, when random obstacles are enabled")
    args = parser.parse_args()
    if not 0.0 < args.himloco_command_filter_alpha <= 1.0:
        parser.error("--himloco-command-filter-alpha must be in (0, 1]")
    if args.stable_pre_roll < 0.0:
        parser.error("--stable-pre-roll must be non-negative")
    np.random.seed(args.seed)
    if args.record and not args.viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    mujoco_viewer = None
    if args.viewer and not args.no_viewer:
        import mujoco.viewer as mujoco_viewer
    model = mujoco.MjModel.from_xml_path(str(SCENES[args.scene]))
    _apply_contact_ab(model, args)
    if args.contact_friction is not None:
        _set_contact_friction(model, args.contact_friction)
    data = mujoco.MjData(model)
    spawn_xy = (0.0, 0.0)
    if args.scene in ("winding_course", "winding_flat_turn"):
        spawn_xy = tuple(model.site("winding_spawn").pos[:2])
    _reset(model, data, spawn_xy=spawn_xy)
    waypoint_path = args.waypoints
    if not waypoint_path and args.scene == "mixed_course":
        default_waypoints = Path("configs/mixed_course_waypoints.json")
        if default_waypoints.exists():
            waypoint_path = str(default_waypoints)
    waypoints = (WaypointManager.from_json(waypoint_path, args.goal_radius)
                 if waypoint_path else
                 WaypointManager.single(args.goal_x, args.goal_y, args.goal_radius))
    obstacle_layout = []
    if args.scene in ("mixed_course", "winding_course", "winding_flat_turn") and args.random_obstacles:
        obstacle_layout = _configure_random_obstacles(
            model, data, args.seed, waypoints, args.random_obstacle_count)
    adapter = MuJoCoStateAdapter(
        model, data, waypoints.current.xy, _terrain(args.scene), args.ray_mode,
        args.angular_velocity_source,
    )
    records = []
    started = time.perf_counter()
    log_handle = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        if args.navigation_policy:
            log_handle = open(args.log, "w", encoding="utf-8")
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    recorder = _Recorder(args.record, model) if args.record else None
    viewer_context = contextlib.nullcontext(None)
    if args.viewer and not args.no_viewer:
        viewer_context = mujoco_viewer.launch_passive(model, data)
    with viewer_context as viewer:
        waypoint_visualizer = None
        if viewer is not None and (args.draw_goal or waypoint_path):
            waypoint_visualizer = WaypointVisualizer(viewer, waypoints.waypoints)
        use_nav = bool(args.navigation_policy)
        if use_nav:
            nav = load_navigation_policy(args.navigation_policy, "", torch.device(args.device))
            him = load_himloco_policy(args.himloco_policy, torch.device(args.device))
            core = PolicyRuntimeCore(
                nav, him, args.device,
                supervisor=TerrainSupervisor(
                    speed_scale=args.speed_scale,
                    command_filter_alpha=args.command_filter_alpha,
                    open_space_assist=not args.no_open_space_assist,
                ),
            )
            control_steps = max(1, int(round(0.02 / model.opt.timestep)))
            output = None
            stopped = False
            waypoint_was_reached = False
            target = np.array([.1, .8, -1.5, -.1, .8, -1.5, .1, 1, -1.5, -.1, 1, -1.5])
            for step in range(args.steps):
                if step % control_steps == 0:
                    state = adapter.read(waypoints.current.xy)
                    waypoint_was_reached = waypoints.update(state.position_xy)
                    if waypoint_was_reached and waypoints.done and args.stop_on_goal:
                        output = core.stop_output()
                        stopped = True
                    else:
                        if waypoint_was_reached:
                            state = adapter.read(waypoints.current.xy)
                        output = core.step(state)
                    action = np.clip(output.himloco_action, -HIMLOCO_ACTION_CLIP, HIMLOCO_ACTION_CLIP)
                    action_scaled = action * .25
                    action_scaled[HIMLOCO_HIP_INDICES] *= HIMLOCO_HIP_REDUCTION
                    target = np.array([.1, .8, -1.5, -.1, .8, -1.5, .1, 1, -1.5, -.1, 1, -1.5]) + action_scaled
                    if stopped:
                        target = np.array([.1, .8, -1.5, -.1, .8, -1.5, .1, 1, -1.5, -.1, 1, -1.5])
                q, dq = data.qpos[7:19], data.qvel[6:18]
                data.ctrl[:] = _himloco_pd_torque(target, q, dq)
                mujoco.mj_step(model, data)
                if output is not None and step % control_steps == 0:
                    record = {"step": step, "raw_command": output.raw_command.tolist(),
                              "supervised_command": output.supervised_command.tolist(),
                              "himloco_action": output.himloco_action.tolist(),
                              "supervisor_state": output.supervisor_state,
                              "control_hz": output.control_hz,
                              "position_xy": state.position_xy.tolist(), "roll": state.roll,
                              "pitch": state.pitch, "min_obstacle_distance": state.min_obstacle_distance,
                              "collision": state.collision, "fallen": state.fallen,
                              "ray_mode": args.ray_mode,
                              "waypoint_index": waypoints.current_index,
                              "waypoints_total": waypoints.total,
                              "target_xy": waypoints.current.xy.tolist(),
                              "target_relative_xy": state.goal_xy.tolist(),
                              "target_distance": waypoints.distance(state.position_xy),
                              "waypoint_reached": waypoint_was_reached,
                              "goal_reached": waypoints.done,
                              "random_obstacles": obstacle_layout}
                    records.append(record)
                    if log_handle is not None:
                        log_handle.write(json.dumps(record) + "\n")
                        log_handle.flush()
                if stopped and args.stop_on_goal:
                    break
                if viewer is not None:
                    if waypoint_visualizer is not None and output is not None:
                        waypoint_visualizer.sync(
                            waypoints.current_index,
                            waypoints.distance(state.position_xy),
                            output.supervisor_state,
                            output.raw_command,
                            output.supervised_command,
                            waypoints.done,
                        )
                    viewer.sync()
                if recorder is not None and step % control_steps == 0:
                    recorder.write(model, data)
        else:
            records = _run_himloco(model, data, args, adapter, viewer, recorder)
    if recorder is not None:
        recorder.close()
    if log_handle is not None:
        log_handle.close()
    elif args.log:
        with open(args.log, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    elapsed = time.perf_counter() - started
    state = adapter.read(waypoints.current.xy)
    positions = np.asarray([r["position_xy"] for r in records if "position_xy" in r], dtype=np.float32)
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0
    max_roll = max((abs(float(r["roll"])) for r in records if "roll" in r), default=abs(state.roll))
    max_pitch = max((abs(float(r["pitch"])) for r in records if "pitch" in r), default=abs(state.pitch))
    min_distance = min((float(r["min_obstacle_distance"]) for r in records if "min_obstacle_distance" in r), default=state.min_obstacle_distance)
    collision = any(r.get("collision", False) for r in records) or state.collision
    fallen = any(r.get("fallen", False) for r in records) or state.fallen
    summary = {"scene": args.scene, "seed": args.seed, "steps": args.steps,
               "elapsed_s": elapsed, "goal_reached": waypoints.done,
               "fallen": fallen, "collision": collision,
               "terrain": state.terrain_hint, "path_length": path_length,
               "max_roll": max_roll, "max_pitch": max_pitch,
               "min_obstacle_distance": min_distance,
               "ray_mode": args.ray_mode,
               "angular_velocity_source": args.angular_velocity_source,
               "waypoints_total": waypoints.total,
               "waypoints_reached": len(waypoints.reached_indices),
               "current_waypoint_index": waypoints.current_index,
               "random_obstacles": obstacle_layout,
               "stop_on_goal": args.stop_on_goal}
    print(json.dumps(summary, sort_keys=True))


class _Recorder:
    def __init__(self, path, model):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("--record requires ffmpeg in PATH")
        self.path = path
        self.process = subprocess.Popen([
            ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "640x480", "-r", "50", "-i", "-",
            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", path,
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.renderer = __import__("mujoco").Renderer(model, height=480, width=640)

    def write(self, model, data):
        # The checked-in scenes intentionally have no named camera; -1 uses
        # MuJoCo's free camera and works for both original and custom XMLs.
        self.renderer.update_scene(data, camera=-1)
        self.process.stdin.write(self.renderer.render().tobytes())

    def close(self):
        self.renderer.close()
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace")
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed while writing {self.path}: {stderr[-500:]}")


if __name__ == "__main__":
    main()

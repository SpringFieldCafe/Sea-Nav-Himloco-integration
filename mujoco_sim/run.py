import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from deploy.go2_onboard.himloco_observation import HIMLocoObservation
from deploy.go2_onboard.model_loader import infer, load_himloco_policy, load_navigation_policy

from .adapters import MuJoCoStateAdapter
from .core import PolicyRuntimeCore


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "go2"
SCENES = {
    "himloco_flat": ASSETS / "scene.xml",
    "stairs_up": ASSETS / "stair.xml",
    "stairs_down": ASSETS / "stairs_down.xml",
    "rough": ASSETS / "hfield.xml",
    "flat_obstacle": ASSETS / "course_flat_obstacle.xml",
    "mixed_course": ASSETS / "course_mixed.xml",
}


def _terrain(scene):
    if scene != "mixed_course":
        return {"stairs_up": "stairs_up", "stairs_down": "stairs_down",
                "rough": "rough"}.get(scene, "flat")
    def mixed_hint(x):
        if 4.5 <= x < 6.2:
            return "stairs_up"
        if 6.2 <= x < 9.8:
            return "rough"
        if 11.5 <= x < 13.0:
            return "stairs_down"
        return "flat"
    return mixed_hint


def _reset(model, data):
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qpos[2] = .45
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:19] = [.1, .8, -1.5, -.1, .8, -1.5, .1, 1, -1.5, -.1, 1, -1.5]
    __import__("mujoco").mj_forward(model, data)


def _run_himloco(model, data, args, adapter, viewer=None, recorder=None):
    policy = load_himloco_policy(args.himloco_policy, torch.device(args.device))
    obs = HIMLocoObservation(torch.device(args.device))
    default = np.array([.1, .8, -1.5, -.1, .8, -1.5, .1, 1, -1.5, -.1, 1, -1.5], dtype=np.float32)
    action = np.zeros(12, dtype=np.float32)
    command = np.asarray([args.vx, args.vy, args.wz], dtype=np.float32)
    records = []
    control_steps = max(1, int(round(.02 / model.opt.timestep)))
    for step in range(args.steps):
        q = data.qpos[7:19]
        dq = data.qvel[6:18]
        data.ctrl[:] = 20 * (default + action * .25 - q) - .5 * dq
        __import__("mujoco").mj_step(model, data)
        if step % control_steps == 0:
            state = adapter.read([args.goal_x, args.goal_y])
            torch_state = {k: torch.from_numpy(v).reshape(1, -1).to(args.device) for k, v in
                           {"command": command, "angular": state.angular_velocity,
                            "gravity": state.gravity, "q": state.joint_position - default,
                            "dq": state.joint_velocity}.items()}
            inp = obs.build(torch_state["command"], torch_state["angular"], torch_state["gravity"], torch_state["q"], torch_state["dq"])
            action = infer(policy, inp, 12)[0].cpu().numpy()
            obs.record_action(torch.from_numpy(action).reshape(1, -1))
            records.append({"step": step, "himloco_action": action.tolist(), "terrain": state.terrain_hint,
                            "position_xy": state.position_xy.tolist(), "roll": state.roll,
                            "pitch": state.pitch, "min_obstacle_distance": state.min_obstacle_distance,
                            "collision": state.collision, "fallen": state.fallen})
        if viewer is not None:
            viewer.sync()
        if recorder is not None and step % control_steps == 0:
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
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--record", default="")
    parser.add_argument("--log", default="")
    args = parser.parse_args()
    np.random.seed(args.seed)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(SCENES[args.scene]))
    data = mujoco.MjData(model)
    _reset(model, data)
    adapter = MuJoCoStateAdapter(model, data, [args.goal_x, args.goal_y], _terrain(args.scene))
    records = []
    started = time.perf_counter()
    recorder = _Recorder(args.record, model) if args.record else None
    viewer_context = contextlib.nullcontext(None)
    if args.viewer and not args.no_viewer:
        viewer_context = mujoco.viewer.launch_passive(model, data)
    with viewer_context as viewer:
        use_nav = bool(args.navigation_policy)
        if use_nav:
            nav = load_navigation_policy(args.navigation_policy, "", torch.device(args.device))
            him = load_himloco_policy(args.himloco_policy, torch.device(args.device))
            core = PolicyRuntimeCore(nav, him, args.device)
            for step in range(args.steps):
                state = adapter.read([args.goal_x, args.goal_y])
                output = core.step(state)
                q, dq = data.qpos[7:19], data.qvel[6:18]
                target = np.array([.1, .8, -1.5, -.1, .8, -1.5, .1, 1, -1.5, -.1, 1, -1.5]) + output.himloco_action * .25
                data.ctrl[:] = 20 * (target - q) - .5 * dq
                mujoco.mj_step(model, data)
                if step % 10 == 0:
                    records.append({"step": step, "raw_command": output.raw_command.tolist(),
                                    "supervised_command": output.supervised_command.tolist(),
                                    "himloco_action": output.himloco_action.tolist(),
                                    "supervisor_state": output.supervisor_state,
                                    "control_hz": output.control_hz,
                                    "position_xy": state.position_xy.tolist(), "roll": state.roll,
                                    "pitch": state.pitch, "min_obstacle_distance": state.min_obstacle_distance,
                                    "collision": state.collision, "fallen": state.fallen})
                if viewer is not None:
                    viewer.sync()
                if recorder is not None and step % 10 == 0:
                    recorder.write(model, data)
        else:
            records = _run_himloco(model, data, args, adapter, viewer, recorder)
    if recorder is not None:
        recorder.close()
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    elapsed = time.perf_counter() - started
    state = adapter.read([args.goal_x, args.goal_y])
    positions = np.asarray([r["position_xy"] for r in records if "position_xy" in r], dtype=np.float32)
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0
    max_roll = max((abs(float(r["roll"])) for r in records if "roll" in r), default=abs(state.roll))
    max_pitch = max((abs(float(r["pitch"])) for r in records if "pitch" in r), default=abs(state.pitch))
    min_distance = min((float(r["min_obstacle_distance"]) for r in records if "min_obstacle_distance" in r), default=state.min_obstacle_distance)
    collision = any(r.get("collision", False) for r in records) or state.collision
    fallen = any(r.get("fallen", False) for r in records) or state.fallen
    summary = {"scene": args.scene, "seed": args.seed, "steps": args.steps,
               "elapsed_s": elapsed, "goal_reached": float(np.linalg.norm(state.goal_xy)) < .45,
               "fallen": fallen, "collision": collision,
               "terrain": state.terrain_hint, "path_length": path_length,
               "max_roll": max_roll, "max_pitch": max_pitch,
               "min_obstacle_distance": min_distance}
    print(json.dumps(summary, sort_keys=True))


class _Recorder:
    def __init__(self, path, model):
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("--record requires imageio") from exc
        self.writer = imageio.get_writer(path, fps=50)
        self.renderer = __import__("mujoco").Renderer(model, height=480, width=640)

    def write(self, model, data):
        self.renderer.update_scene(data, camera="track")
        self.writer.append_data(self.renderer.render())

    def close(self):
        self.renderer.close()
        self.writer.close()


if __name__ == "__main__":
    main()

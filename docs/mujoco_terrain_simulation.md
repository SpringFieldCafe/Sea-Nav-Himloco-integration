# MuJoCo terrain simulation

This branch keeps the deployed Go2 HIMLoco contract unchanged and uses the
MuJoCo scenes from the original HIMLoco project as the reference terrain
implementation. The checked-in simulator assets are a local copy of those
scene files and meshes; policies remain separate files and are never retrained
or re-exported by the simulator.

Navigation commands below use `sea_nav_policy_peer_model_2000.pt`, exported
from `models/navigation/model_2000.pt`, the peer-trained PPO checkpoint used
by the successful Isaac Gym run. The older `sea_nav_policy_2000.pt` artifact is
kept for comparison and is not the recommended policy for this branch.

The mixed course uses a tiled rough field whose center row is `y=0`; the
waypoints are placed at the tile centers `x=0,4,8,12,16,20` rather than on
tile seams. Optional extra boxes can be generated at runtime with a seeded
layout sampler. The default layout uses 10 moderate boxes. The same seed reproduces the same layout; changing `--seed`
creates a different layout while preserving a clear center corridor around the
waypoints.

The mixed course intentionally does not use the old long `narrow_left` and
`narrow_right` wall pair. Those 4 m walls create a corridor unlike the random
box training distribution and can dominate the navigation behavior.

## HIMLoco contract

- input: `(N, 270)` = six newest-to-oldest frames of 45 values;
- output: `(N, 12)` joint actions;
- control: 50 Hz (`0.002 s` physics step, decimation 10);
- action scale: `0.25`; PD: `Kp=20`, `Kd=0.5`;
- previous action is clipped to `[-100, 100]` before it enters the next frame;
- policy order is `FL, FR, RL, RR`, while Go2 motor order is converted by the
  existing `deploy.go2_onboard.joint_mapping` contract.

## Original terrain recovery

From the repository root, with MuJoCo and the project dependencies installed:

```bash
python -m mujoco_sim.run --scene himloco_flat --himloco-policy models/locomotion/himloco/policy_1.pt --viewer
python -m mujoco_sim.run --scene stairs_up --himloco-policy models/locomotion/himloco/policy_1.pt --viewer
python -m mujoco_sim.run --scene stairs_down --himloco-policy models/locomotion/himloco/policy_1.pt --viewer
python -m mujoco_sim.run --scene rough --himloco-policy models/locomotion/himloco/policy_1.pt --viewer
```

`stairs_up`, `stairs_down`, and `rough` reuse the original `stair.xml` and
`hfield.xml` terrain definitions. `--steps`, `--seed`, `--record`, and
`--no-viewer` are available for headless smoke tests and video capture.

## SEA-Nav and mixed course

The mixed course uses the ordered waypoint file
`configs/mixed_course_waypoints.json` by default. Waypoints are visualized in
the passive viewer, become the existing `relative_goal_xy` field, and stop the
episode at the final point when `--stop-on-goal` is supplied. The ray mode
defaults to `grid2ray`, which is the Isaac-compatible terrain-truth mode; use
`physical_lidar` for the MuJoCo geometry-ray comparison.

```bash
python -m mujoco_sim.run --scene flat_obstacle --navigation-policy artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt --himloco-policy models/locomotion/himloco/policy_1.pt --goal-x 12 --goal-y 0 --no-viewer
python -m mujoco_sim.run --scene mixed_course --navigation-policy artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt --himloco-policy models/locomotion/himloco/policy_1.pt --goal-x 17 --goal-y 0 --viewer --record logs/mixed_course.mp4
```

For ordered waypoint evaluation:

```bash
python -m mujoco_sim.run --scene mixed_course --navigation-policy artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt --himloco-policy models/locomotion/himloco/policy_1.pt --waypoints configs/mixed_course_waypoints.json --ray-mode grid2ray --goal-radius 0.6 --stop-on-goal --viewer --log logs/mixed_course_grid2ray.jsonl --record logs/mixed_course_grid2ray.mp4
python -m mujoco_sim.run --scene mixed_course --navigation-policy artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt --himloco-policy models/locomotion/himloco/policy_1.pt --waypoints configs/mixed_course_waypoints.json --ray-mode physical_lidar --goal-radius 0.6 --stop-on-goal --no-viewer --log logs/mixed_course_lidar.jsonl

# Seed-reproducible randomized obstacles in the rough course.
python -m mujoco_sim.run --scene mixed_course --navigation-policy artifacts/go2_onboard/sea_nav_policy_peer_model_2000.pt --himloco-policy models/locomotion/himloco/policy_1.pt --waypoints configs/mixed_course_waypoints.json --ray-mode grid2ray --random-obstacles --random-obstacle-count 10 --seed 0 --speed-scale 1.5 --command-filter-alpha 0.7 --viewer --draw-goal --log logs/mixed_course_random_seed0.jsonl
```

SEA-Nav uses the existing 55-value frame and 10-frame history, including 41
projected rays, then clips commands to `vx [-1,1]`, `vy [-1,1]`, `wz [-2,2]`.
The terrain supervisor applies stricter limits on stairs and rough terrain.

## Evaluation output

Each run prints a JSON summary and writes one JSONL record per control cycle
when `--log` is supplied. The records include waypoint index and target,
relative goal, goal distance, ray mode, waypoint/goal completion, and
goal/fall/collision status,
terrain completion, elapsed time, path length, minimum obstacle distance,
maximum roll/pitch, raw and supervised SEA-Nav commands, HIMLoco actions, and
measured control frequency.

## Common problems

- **MuJoCo import error:** install the environment's pinned `mujoco` package;
  the simulator does not silently fall back to another physics engine.
- **Missing policy:** pass a repository-relative or absolute path explicitly;
  no machine-specific path is embedded in the code.
- **Robot falls immediately:** first run `himloco_flat` and verify the policy
  is `policy_1.pt`, not a navigation checkpoint. Then validate stairs and rough
  terrain independently before `mixed_course`.
- **Viewer unavailable:** use `--no-viewer`; headless runs still produce the
  same metrics and smoke-test checks.
- **Unexpected action order:** do not edit XML actuator order. The adapter
  resolves named joints and applies the checked-in policy/motor permutation.

## Required debug order

`himloco_flat` -> `stairs_up` -> `stairs_down` -> `rough` ->
`flat_obstacle` -> `approach_stairs` -> `align_stairs` -> `stairs_up_closed_loop`
-> `stairs_down_closed_loop` -> `mixed_course`.

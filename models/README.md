# Shared model bundle

This directory keeps the two policy roles separate:

```text
models/
├── locomotion/
│   └── himloco/
│       └── policy_1.pt
└── navigation/
    └── peer_policy.pt
```

## Locomotion policy

`locomotion/himloco/policy_1.pt` is the HIMLoco-Go2 rough-terrain exported
stacked actor. It expects a 270-dimensional input made from six newest-to-oldest
45-dimensional observation frames and returns 12 joint actions. It is a
low-level locomotion policy; it is not the SEA-Nav navigation policy.

Run a direct viewer smoke test from the repository root:

```bash
python training/legged_gym/legged_gym/tests/test_env.py \
  --task go2_pos_rough \
  --locomotion_backend himloco \
  --himloco_policy models/locomotion/himloco/policy_1.pt \
  --viewer \
  --smoke_steps 2000 \
  --smoke_command 0.2 0 0
```

## Navigation policy

Place the peer-trained high-level navigation policy at:

```text
models/navigation/peer_policy.pt
```

The navigation policy must produce the SEA-Nav navigation action
`[vx, vy, wz]`. Its exact loader arguments depend on the training run and
checkpoint format, so the file is not treated as a HIMLoco actor.

For a complete PPO checkpoint such as `model_2000.pt`, evaluate it with:

```bash
python training/legged_gym/legged_gym/scripts/play.py \
  --task go2_pos_rough \
  --navigation_checkpoint models/navigation/model_2000.pt \
  --locomotion_backend himloco \
  --himloco_policy models/locomotion/himloco/policy_1.pt
```

The two model files are intentionally independent. The navigation policy
selects commands, while the HIMLoco policy turns those commands into 12 joint
actions. No HIMLoco source repository is copied into this project.

Navigation model files are ignored by Git. Keep the local checkpoint at the
documented path or pass another path explicitly; do not commit checkpoint
contents or the extracted serialization directory.

# SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments


**Project Website**: [https://11chens.github.io/sea-nav](https://11chens.github.io/sea-nav/)

<p align="center">
  <img src="imgs/terser.jpg" width="80%">
</p>

---

## Installation

### 1. Environment Setup
Create a new Python virtual environment with Python 3.8:
```bash
conda create -n sea_nav python=3.8
conda activate sea_nav
```

### 2. Install Isaac Gym
- Download and install Isaac Gym Preview 4 from [NVIDIA Developer](https://developer.nvidia.com/isaac-gym).
- Install the python package:
```bash
cd isaacgym/python && pip install -e .
```

### 3. Install rsl_rl
- Clone this repository
- Install the package:
```bash
cd training/rsl_rl && pip install -e .
```

### 4. Install legged_gym
```bash
cd training/legged_gym && pip install -e .
```

---

## Usage

### Training
To start training in headless mode:
```bash
python training/legged_gym/legged_gym/scripts/train.py --headless
```

### Testing
To visualize and test a trained policy:
```bash
python training/legged_gym/legged_gym/scripts/play.py
```

### HIMLoco locomotion backend

SEA-Nav keeps the original SLR backend as the default. HIMLoco can be selected
at runtime without copying its repository or policy into SEA-Nav:

```bash
python training/legged_gym/legged_gym/tests/test_env.py \
  --task go2_pos_rough \
  --locomotion_backend himloco \
  --himloco_policy /path/to/HIMLoco/legged_gym/logs/rough_go2/exported/policies/policy_1.pt \
  --viewer \
  --smoke_steps 2000 \
  --smoke_command 0.2 0 0
```

The HIMLoco policy path is a command-line input and is intentionally not
hard-coded. The current adapter expects the rough-go2 exported stacked actor:
270 input dimensions (45 observations over 6 newest-to-oldest history frames)
and 12 joint outputs. The policy uses its own action scale, default joint
angles, PD gains, command bounds, and action clipping defined by the HIMLoco
contract in the Go2 configuration.

The `policy_1.pt` file is a locomotion policy, not the SEA-Nav navigation
policy. The high-level navigation policy remains loaded separately by the
normal evaluation workflow. `--smoke_command VX VY WZ` is only for direct
locomotion smoke tests; the trained HIMLoco command ranges are `vx,vy ∈ [-1,1]`
and `wz ∈ [-2,2]`.

The SLR backend remains available with the default configuration:

```bash
python -m pytest -q \
  training/legged_gym/legged_gym/tests/test_himloco_policy.py \
  training/legged_gym/legged_gym/tests/test_himloco_contract.py \
  training/legged_gym/legged_gym/tests/test_slr_backend_regression.py
```

For a shareable two-policy layout, see [`models/README.md`](models/README.md).
The bundled HIMLoco policy is the low-level locomotion actor; the peer-trained
navigation policy belongs in `models/navigation/peer_policy.pt` and must still
be supplied separately until its exported file is available.

To evaluate a local SEA-Nav PPO checkpoint directly with HIMLoco:

```bash
python training/legged_gym/legged_gym/scripts/play.py \
  --task go2_pos_rough \
  --navigation_checkpoint models/navigation/model_2000.pt \
  --locomotion_backend himloco \
  --himloco_policy models/locomotion/himloco/policy_1.pt \
  --navigation_speed_scale 1.2
```

`--navigation_checkpoint` takes the complete PPO checkpoint and loads only
its model state for inference; the optimizer state is not used. The optional
`--navigation_speed_scale` multiplies the policy's forward `vx` command during
visual play; `1.0` preserves the original policy output.

---

## Deployment (Coming soon)
For instructions on deploying to real-world robots, please refer to the [deployment README](deployment/README.md).

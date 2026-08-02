import os

import pytest
import torch


POLICY_ENV_VAR = "HIMLOCO_POLICY"


def test_himloco_policy_batch_shapes():
    policy_path = os.environ.get(POLICY_ENV_VAR)
    if not policy_path:
        pytest.skip(f"set {POLICY_ENV_VAR} to run the offline HIMLoco policy test")

    policy = torch.jit.load(policy_path, map_location="cpu").eval()
    with torch.inference_mode():
        for batch_size in (1, 4):
            output = policy(torch.zeros(batch_size, 270, dtype=torch.float32))
            assert tuple(output.shape) == (batch_size, 12)
            assert torch.isfinite(output).all()

        random_output = policy(torch.randn(4, 270, dtype=torch.float32))
        assert tuple(random_output.shape) == (4, 12)
        assert torch.isfinite(random_output).all()

import os
import importlib.util
import importlib
import glob

import pytest
import isaacgym  # noqa: F401
import torch


POLICY_ENV_VAR = "HIMLOCO_POLICY"


def _reuse_cached_gymtorch():
    """Avoid rebuilding Isaac Gym's already-built extension during pytest collection."""
    import sys

    if "isaacgym.gymtorch" in sys.modules:
        return

    import torch.utils.cpp_extension as cpp_extension

    extension_root = os.environ.get(
        "TORCH_EXTENSIONS_DIR", cpp_extension.get_default_build_root()
    )
    candidates = glob.glob(
        os.path.join(extension_root, "**", "gymtorch", "gymtorch.so"),
        recursive=True,
    )
    cuda_tag = "cu" + str(torch.version.cuda).replace(".", "")
    matching = [path for path in candidates if cuda_tag in path]
    extension_path = matching[0] if matching else (candidates[0] if candidates else None)
    if extension_path is None:
        return

    def cached_load(*args, **kwargs):
        spec = importlib.util.spec_from_file_location("gymtorch", extension_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    original_load = cpp_extension.load
    cpp_extension.load = cached_load
    try:
        importlib.import_module("isaacgym.gymtorch")
    finally:
        cpp_extension.load = original_load


_reuse_cached_gymtorch()


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

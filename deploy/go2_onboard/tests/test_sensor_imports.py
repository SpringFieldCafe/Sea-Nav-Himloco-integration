import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_sensor_modules_import_without_torch():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PATH"] = "/usr/bin:/bin"
    env.pop("PYTHONHOME", None)
    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            "import deploy.go2_onboard.diagnostics, deploy.go2_onboard.runtime, deploy.go2_onboard.sensor_bridge",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_shadow_reports_missing_torch():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PATH"] = "/usr/bin:/bin"
    env.pop("PYTHONHOME", None)
    code = (
        "from types import SimpleNamespace; "
        "from deploy.go2_onboard.runtime import OnboardRuntime; "
        "runtime = object.__new__(OnboardRuntime); "
        "runtime.args = SimpleNamespace(navigation_policy='', navigation_metadata='', himloco_policy='', device='cpu'); "
        "runtime._load_policies()"
    )
    result = subprocess.run(
        ["/usr/bin/python3", "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "shadow mode requires torch" in result.stderr

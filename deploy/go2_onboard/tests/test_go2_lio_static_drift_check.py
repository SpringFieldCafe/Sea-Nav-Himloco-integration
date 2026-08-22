import importlib.util


def load_recorder():
    spec = importlib.util.spec_from_file_location(
        "go2_lio_static_drift_check",
        "tools/go2_lio_static_drift_check.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mean_std_and_wrapped_yaw_delta():
    recorder = load_recorder()
    mean, std = recorder.mean_std([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert round(std, 9) == round((2.0 / 3.0) ** 0.5, 9)
    assert abs(recorder.angle_delta(-3.13, 3.13)) < 0.03

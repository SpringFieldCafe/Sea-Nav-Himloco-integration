import importlib.util
import struct
from types import SimpleNamespace

import pytest


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "go2_lio_time_contract_check",
        "tools/go2_lio_time_contract_check.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_cloud(values):
    checker = load_checker()
    field = SimpleNamespace(name="time", datatype=7, offset=12, count=1)
    data = b"".join(struct.pack("<ffff", 1.0, 2.0, 3.0, value) for value in values)
    message = SimpleNamespace(
        fields=[field],
        width=len(values),
        height=1,
        point_step=16,
        row_step=16 * len(values),
        is_bigendian=False,
        data=data,
    )
    return checker, message


def test_sampled_point_time_field_reads_binary_values():
    checker, message = fake_cloud([0.0, 0.02, 0.04, 0.06])
    values = checker.sampled_point_field_values(message, "time", max_samples=4)
    assert values == pytest.approx([0.0, 0.02, 0.04, 0.06])


def test_time_unit_inference_uses_scan_period():
    checker = load_checker()
    assert checker.infer_time_unit([0.06, 0.065], 0.067)[0] == "SECONDS"
    assert checker.infer_time_unit([60.0, 65.0], 0.067)[0] == "MILLISECONDS"
    assert checker.infer_time_unit([60000.0, 65000.0], 0.067)[0] == "MICROSECONDS"
    assert checker.infer_time_unit([60000000.0, 65000000.0], 0.067)[0] == "NANOSECONDS"


def test_monotonic_and_nearest_offset_helpers():
    checker = load_checker()
    assert checker.monotonic([1.0, 1.0, 2.0]) is True
    assert checker.monotonic([1.0, 0.9]) is False
    assert checker.nearest_absolute_offsets([1.0, 2.0], [0.99, 2.01]) == pytest.approx([0.01, 0.01])


def test_ordered_offsets_keep_a_shared_transform_offset():
    checker = load_checker()
    raw_cloud = [10.0, 10.1, 10.2]
    transformed_cloud = [19.0, 19.1, 19.2]
    raw_imu = [10.0, 10.01, 10.02]
    transformed_imu = [19.0, 19.01, 19.02]
    assert checker.ordered_signed_offsets(raw_cloud, transformed_cloud) == pytest.approx([9.0, 9.0, 9.0])
    assert checker.ordered_signed_offsets(raw_imu, transformed_imu) == pytest.approx([9.0, 9.0, 9.0])
    assert checker.combined_check("PASS", "PASS") == "PASS"

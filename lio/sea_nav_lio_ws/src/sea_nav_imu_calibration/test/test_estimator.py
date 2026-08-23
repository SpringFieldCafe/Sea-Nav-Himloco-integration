import csv
import random

from sea_nav_imu_calibration.estimate_calibration import estimate, load_records


FIELDS = [
    'seq_local', 'receive_monotonic_time', 'header_stamp_sec',
    'header_stamp_nanosec', 'frame_id', 'orientation_x', 'orientation_y',
    'orientation_z', 'orientation_w', 'angular_velocity_x',
    'angular_velocity_y', 'angular_velocity_z', 'linear_acceleration_x',
    'linear_acceleration_y', 'linear_acceleration_z',
]


def _raw_from_transformed(values):
    return values


def test_estimator_recovers_cmu_formula(tmp_path):
    random.seed(43)
    acc_bias = (0.10, -0.20, 0.30)
    ang_bias = (0.01, -0.02, 0.03)
    projection = (0.12, -0.08)
    path = tmp_path / 'synthetic_imu.csv'

    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(2200):
            t = index * 0.01
            rotating = t >= 10.0
            target_gyro = (
                ang_bias[0] - projection[0] if rotating else ang_bias[0],
                ang_bias[1] - projection[1] if rotating else ang_bias[1],
                ang_bias[2] + 1.0 if rotating else ang_bias[2],
            )
            target_acc = (acc_bias[0], acc_bias[1], 9.81 + acc_bias[2])
            gyro = _raw_from_transformed(target_gyro)
            acc = _raw_from_transformed(target_acc)
            noise = lambda: random.gauss(0.0, 1e-5)
            writer.writerow({
                'seq_local': index,
                'receive_monotonic_time': '%.6f' % t,
                'header_stamp_sec': 0,
                'header_stamp_nanosec': index * 10000000,
                'frame_id': 'utlidar_imu',
                'orientation_x': 0.0,
                'orientation_y': 0.0,
                'orientation_z': 0.0,
                'orientation_w': 1.0,
                'angular_velocity_x': gyro[0] + noise(),
                'angular_velocity_y': gyro[1] + noise(),
                'angular_velocity_z': gyro[2] + noise(),
                'linear_acceleration_x': acc[0] + noise(),
                'linear_acceleration_y': acc[1] + noise(),
                'linear_acceleration_z': acc[2] + noise(),
            })

    result = estimate(load_records(str(path)), 1.0, 9.0, 10.0, 21.0)
    assert all(abs(actual - expected) < 2e-3 for actual, expected in zip(result['acc_bias'], acc_bias))
    assert all(abs(actual - expected) < 2e-3 for actual, expected in zip(result['ang_bias'], ang_bias))
    assert all(abs(actual - expected) < 2e-3 for actual, expected in zip(result['projection'], projection))
    assert result['rotation_signal_valid']

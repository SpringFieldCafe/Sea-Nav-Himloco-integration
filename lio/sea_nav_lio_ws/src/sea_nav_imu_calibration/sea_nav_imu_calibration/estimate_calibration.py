#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys


THETA = math.radians(15.1)
REQUIRED_FIELDS = {
    'receive_monotonic_time',
    'angular_velocity_x',
    'angular_velocity_y',
    'angular_velocity_z',
    'linear_acceleration_x',
    'linear_acceleration_y',
    'linear_acceleration_z',
}


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('%s is not finite' % name)
    return value


def _transform(row):
    x = _finite(row['angular_velocity_x'], 'angular_velocity_x')
    y = -_finite(row['angular_velocity_y'], 'angular_velocity_y')
    z = -_finite(row['angular_velocity_z'], 'angular_velocity_z')
    angular_x = x * math.cos(THETA) - z * math.sin(THETA)
    angular_y = y
    angular_z = x * math.sin(THETA) + z * math.cos(THETA)

    acc_x = _finite(row['linear_acceleration_x'], 'linear_acceleration_x')
    acc_y = -_finite(row['linear_acceleration_y'], 'linear_acceleration_y')
    acc_z = -_finite(row['linear_acceleration_z'], 'linear_acceleration_z')
    linear_x = acc_x * math.cos(THETA) - acc_z * math.sin(THETA)
    linear_y = acc_y
    linear_z = acc_x * math.sin(THETA) + acc_z * math.cos(THETA)
    return {
        'time': _finite(row['receive_monotonic_time'], 'receive_monotonic_time'),
        'angular': (angular_x, angular_y, angular_z),
        'linear': (linear_x, linear_y, linear_z),
    }


def load_records(path):
    if not os.path.isfile(path):
        raise ValueError('input CSV does not exist: %s' % path)
    with open(path, newline='') as input_file:
        reader = csv.DictReader(input_file)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_FIELDS - fields
        if missing:
            raise ValueError('input CSV missing fields: %s' % ', '.join(sorted(missing)))
        records = [_transform(row) for row in reader]
    if not records:
        raise ValueError('input CSV contains no samples')
    previous = records[0]['time']
    for record in records[1:]:
        if record['time'] <= previous:
            raise ValueError('receive_monotonic_time is not strictly increasing')
        previous = record['time']
    return records


def _window(records, start, end):
    if end <= start:
        raise ValueError('window end must be greater than start')
    origin = records[0]['time']
    return [record for record in records if start <= record['time'] - origin <= end]


def _mean(records, key, index):
    return sum(record[key][index] for record in records) / len(records)


def _std(records, key, index, mean):
    return math.sqrt(sum((record[key][index] - mean) ** 2 for record in records) / len(records))


def estimate(records, static_start, static_end, rotate_start, rotate_end):
    static = _window(records, static_start, static_end)
    rotate = _window(records, rotate_start, rotate_end)
    if not static:
        raise ValueError('static window contains no samples')
    if not rotate:
        raise ValueError('rotation window contains no samples')

    acc_mean = [_mean(static, 'linear', i) for i in range(3)]
    gyro_mean = [_mean(static, 'angular', i) for i in range(3)]
    acc_bias = [acc_mean[0], acc_mean[1], acc_mean[2] - 9.81]
    ang_bias = list(gyro_mean)

    rotate_mean = [_mean(rotate, 'angular', i) for i in range(3)]
    corrected = [rotate_mean[i] - ang_bias[i] for i in range(3)]
    if abs(corrected[2]) < 1e-9:
        raise ValueError('rotation z signal is too small for projection estimate')
    projection = [-corrected[0] / corrected[2], -corrected[1] / corrected[2]]

    static_gyro_std = [_std(static, 'angular', i, gyro_mean[i]) for i in range(3)]
    static_acc_std = [_std(static, 'linear', i, acc_mean[i]) for i in range(3)]
    rotate_gyro_std = [_std(rotate, 'angular', i, rotate_mean[i]) for i in range(3)]
    rotation_signal_valid = (
        abs(corrected[2]) > 0.1 and
        abs(corrected[2]) > max(3.0 * static_gyro_std[2], 1e-6)
    )
    warnings = []
    if static_end - static_start < 5.0:
        warnings.append('static window is shorter than 5 seconds')
    if rotate_end - rotate_start < 10.0:
        warnings.append('rotation window is shorter than 10 seconds')
    if not rotation_signal_valid:
        warnings.append('rotation z signal is weak')

    return {
        'static': static,
        'rotate': rotate,
        'acc_bias': acc_bias,
        'ang_bias': ang_bias,
        'projection': projection,
        'static_gyro_mean': gyro_mean,
        'static_gyro_std': static_gyro_std,
        'static_acc_mean': acc_mean,
        'static_acc_std': static_acc_std,
        'rotate_gyro_mean': rotate_mean,
        'rotate_gyro_std': rotate_gyro_std,
        'rotation_signal_valid': rotation_signal_valid,
        'warnings': warnings,
    }


def _write_yaml(path, result, force):
    if os.path.exists(path) and not force:
        raise ValueError('refusing to overwrite existing output; use --force explicitly')
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    values = {
        'acc_bias_x': result['acc_bias'][0],
        'acc_bias_y': result['acc_bias'][1],
        'acc_bias_z': result['acc_bias'][2],
        'ang_bias_x': result['ang_bias'][0],
        'ang_bias_y': result['ang_bias'][1],
        'ang_bias_z': result['ang_bias'][2],
        'ang_z2x_proj': result['projection'][0],
        'ang_z2y_proj': result['projection'][1],
    }
    with open(path, 'w') as output_file:
        for key, value in values.items():
            output_file.write('%s: %.17g\n' % (key, value))


def _arguments():
    parser = argparse.ArgumentParser(description='Estimate IMU calibration from a recorded CSV.')
    parser.add_argument('input')
    parser.add_argument('--output', default='imu_calibration_candidate.yaml')
    parser.add_argument('--static-start', type=float, required=True)
    parser.add_argument('--static-end', type=float, required=True)
    parser.add_argument('--rotate-start', type=float, required=True)
    parser.add_argument('--rotate-end', type=float, required=True)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args(sys.argv[1:])


def _print_report(input_path, output_path, records, static_start, static_end, rotate_start, rotate_end, result):
    origin = records[0]['time']
    static_duration = result['static'][-1]['time'] - result['static'][0]['time']
    rotate_duration = result['rotate'][-1]['time'] - result['rotate'][0]['time']
    print('INPUT_FILE = %s' % input_path)
    print('TOTAL_SAMPLES = %d' % len(records))
    print('STATIC_SAMPLE_COUNT = %d' % len(result['static']))
    print('STATIC_DURATION = %.6f' % static_duration)
    print('ROTATE_SAMPLE_COUNT = %d' % len(result['rotate']))
    print('ROTATE_DURATION = %.6f' % rotate_duration)
    for key, value in zip(('X', 'Y', 'Z'), result['acc_bias']):
        print('ACC_BIAS_%s = %.9g' % (key, value))
    for key, value in zip(('X', 'Y', 'Z'), result['ang_bias']):
        print('ANG_BIAS_%s = %.9g' % (key, value))
    print('ANG_Z2X_PROJ = %.9g' % result['projection'][0])
    print('ANG_Z2Y_PROJ = %.9g' % result['projection'][1])
    print('STATIC_GYRO_STD = %s' % result['static_gyro_std'])
    print('STATIC_ACC_STD = %s' % result['static_acc_std'])
    print('ROTATE_GYRO_MEAN_X = %.9g' % result['rotate_gyro_mean'][0])
    print('ROTATE_GYRO_MEAN_Y = %.9g' % result['rotate_gyro_mean'][1])
    print('ROTATE_GYRO_MEAN_Z = %.9g' % result['rotate_gyro_mean'][2])
    print('ROTATE_GYRO_STD_Z = %.9g' % result['rotate_gyro_std'][2])
    print('ROTATION_SIGNAL_VALID = %s' % ('YES' if result['rotation_signal_valid'] else 'NO'))
    for warning in result['warnings']:
        print('WARNING = %s' % warning)
    print('OUTPUT_FILE = %s' % output_path)


def main():
    args = _arguments()
    records = load_records(args.input)
    result = estimate(
        records,
        args.static_start,
        args.static_end,
        args.rotate_start,
        args.rotate_end,
    )
    _write_yaml(args.output, result, args.force)
    _print_report(
        args.input,
        args.output,
        records,
        args.static_start,
        args.static_end,
        args.rotate_start,
        args.rotate_end,
        result,
    )


if __name__ == '__main__':
    main()

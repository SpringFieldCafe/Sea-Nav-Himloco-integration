# Offline deskew diagnostic

This directory contains an offline-only diagnostic for the transformed
Point-LIO input bags. It does not change Point-LIO or any runtime parameter.

Generate a deskewed bag:

```bash
/usr/bin/python3 tools/offline_deskew_test/deskew_bag.py \
  --input-bag /tmp/lio_bags/good_candidate/transformed_input \
  --output-bag /tmp/lio_bags/good_candidate_deskew/transformed_input
```

The output keeps `/sea_nav/lio/transformed_raw_imu` unchanged and rewrites
only xyz in `/sea_nav/lio/transformed_cloud`. All original cloud fields,
including `intensity`, `ring`, and `time`, are retained.

For a fair A/B replay, run the same Point-LIO launch twice, once with the
original bag and once with the generated bag. Ensure no live transformer or
live cloud/IMU publisher is running, and use identical replay rate and
Point-LIO parameters.

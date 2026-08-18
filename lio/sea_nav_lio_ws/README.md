# SEA-Nav Go2 Point-LIO workspace

This workspace is independent of `/unitree/module` and contains no motion-control
node. It consumes the existing read-only DDS topics and publishes only derived
odometry and registered clouds.

## Input contract

- `/utlidar/cloud`: `sensor_msgs/msg/PointCloud2`, `ring=0..17`, `scan_line=18`,
  `time` is seconds within the scan, and the point fields are `x/y/z/intensity/ring/time`.
- `/utlidar/imu`: `sensor_msgs/msg/Imu`, approximately 249 Hz.
- RMW during the Go2 run: `rmw_cyclonedds_cpp`.

## Output contract

- `/sea_nav/lio/odom`: Point-LIO's native `aft_mapped_to_init` odometry.
- `/sea_nav/lio/cloud_registered`: Point-LIO's native `cloud_registered` output.

The native Point-LIO frame ids are preserved. A later adapter may translate them
to the SEA-Nav odometry contract after static validation.

## Provenance

The Point-LIO and `transform_sensors` packages are based on CMU's
`autonomy_stack_go2`, branch `foxy-humble`, commit `43d5f54`. The transform node
keeps the upstream calibration and sensor-frame equations. The only source-level
change is removing the duplicate `rclpy.spin(self)` from the constructor; the
module's `main()` remains the sole executor owner.

`mapping/imu_time_inte` is set to `0.0040160643` from the measured 249 Hz input.
Point-LIO uses this value only in its `imu_en=false` covariance propagation path;
with the Go2 configuration's `imu_en=true`, timestamped IMU samples drive the
propagation directly.

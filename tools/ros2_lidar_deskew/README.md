# ROS 2 LiDAR deskew diagnostic

This node is standalone and is not included in any existing launch file.

Start it in a localhost-only replay domain:

```bash
source /opt/ros/humble/setup.bash
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=73
/usr/bin/python3 tools/ros2_lidar_deskew/deskew_node.py
```

It subscribes to transformed cloud/IMU and publishes:

`/sea_nav/lio/deskewed_cloud`

The node preserves all input PointCloud2 fields and only changes xyz. Its
integration is for diagnosis, not a production estimator replacement.

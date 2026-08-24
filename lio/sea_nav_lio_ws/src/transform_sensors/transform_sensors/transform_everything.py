#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import yaml

import os
import math

class Repuber(Node):
    # Unitree's L1 cloud and embedded IMU frames are parallel and co-oriented.
    # Point-LIO owns the L1-to-IMU translation; this node only normalizes the
    # message contract and applies calibration in the native sensor axes.
    FRAME_CONTRACT = 'unilidar_native_sensor_v1'
    RAW_CLOUD_FRAME = 'utlidar_lidar'
    BASE_CLOUD_FRAME = 'base_link'
    RAW_IMU_FRAME = 'utlidar_imu'

    def __init__(self):
        super().__init__('sensor_transformer')
        self.declare_parameter('imu_ang_z2x_proj', float('nan'))
        self.declare_parameter('imu_ang_z2y_proj', float('nan'))
        self.imu_sub = self.create_subscription(Imu, '/utlidar/imu', self.imu_callback, 50)
        self.cloud_sub = self.create_subscription(PointCloud2, '/utlidar/cloud', self.cloud_callback, 50)
        
        self.imu_raw_pub = self.create_publisher(Imu, '/utlidar/transformed_raw_imu', 50)
        self.imu_pub = self.create_publisher(Imu, '/utlidar/transformed_imu', 50)
        self.cloud_pub = self.create_publisher(PointCloud2, '/utlidar/transformed_cloud', 50)
        self.cloud_base_pub = self.create_publisher(PointCloud2, '/utlidar/cloud_base', 50)

        self.imu_stationary_list = []
        
        self.time_stamp_offset = 0
        self.time_stamp_offset_set = False
        
        # Load only calibration produced for this frame contract. A calibration
        # generated before the frame fix must not be silently reused.
        default_calib_data = {
                'acc_bias_x': 0.0,
                'acc_bias_y': 0.0,
                'acc_bias_z': 0.0,
                'ang_bias_x': 0.0,
                'ang_bias_y': 0.0,
                'ang_bias_z': 0.0,
                'ang_z2x_proj': 0.0,
                'ang_z2y_proj': 0.0
            }
        calib_data = default_calib_data
        calib_source = 'legacy_fallback_default'
        calib_file_path = os.path.join(os.path.expanduser('~'), 'Desktop/imu_calib_data.yaml')
        try:
            with open(calib_file_path, 'r') as calib_file:
                candidate = yaml.load(calib_file, Loader=yaml.FullLoader) or {}
            if candidate.get('frame_contract') != self.FRAME_CONTRACT:
                print('IMU_CALIBRATION_INVALIDATED=YES')
                print('IMU_CALIBRATION_WARNING=calibration requires recalibration for unilidar native frame contract')
                print('IMU_CALIBRATION_FILE_IGNORED=%s' % calib_file_path)
            else:
                calib_data = candidate
                calib_source = 'calibration_yaml'
                print("imu_calib.yaml loaded")
        except OSError:
            print("imu_calib.yaml not found, using default values")

        if calib_source != 'calibration_yaml':
            print('NO_VALID_NATIVE_IMU_CALIBRATION')
            
        self.acc_bias_x = calib_data['acc_bias_x']
        self.acc_bias_y = calib_data['acc_bias_y']
        self.acc_bias_z = calib_data['acc_bias_z']
        self.ang_bias_x = calib_data['ang_bias_x']
        self.ang_bias_y = calib_data['ang_bias_y']
        self.ang_bias_z = calib_data['ang_bias_z']
        override_x = float(self.get_parameter('imu_ang_z2x_proj').value)
        override_y = float(self.get_parameter('imu_ang_z2y_proj').value)
        override_x_set = math.isfinite(override_x)
        override_y_set = math.isfinite(override_y)
        self.ang_z2x_proj = override_x if override_x_set else calib_data['ang_z2x_proj']
        self.ang_z2y_proj = override_y if override_y_set else calib_data['ang_z2y_proj']
        if override_x_set or override_y_set:
            if calib_source == 'calibration_yaml':
                calib_source = 'explicit_override_over_calibration_yaml'
            elif override_x_set and override_y_set:
                calib_source = 'explicit_override_over_legacy_defaults'
            else:
                calib_source = 'mixed_explicit_override_and_legacy_or_calibration'

        print(f"IMU_ANG_Z2X_PROJ={self.ang_z2x_proj:.9f}")
        print(f"IMU_ANG_Z2Y_PROJ={self.ang_z2y_proj:.9f}")
        print(f"IMU_CALIB_SOURCE={calib_source}")
                
        self.sensor_rotation = np.eye(3)

        # Used only to preserve the existing robot-body self-filter. It is not
        # applied to the published cloud, whose origin remains the LiDAR.
        self.base_from_lidar_rotation = np.array([
            [-0.965512906, 0.0, 0.260355197],
            [0.0, 1.0, 0.0],
            [-0.260355197, 0.0, -0.965512906],
        ])
        self.base_from_lidar_translation = np.array([0.28945, 0.0, -0.046825])
        self.x_filter_min = -0.7
        self.x_filter_max = -0.1
        self.y_filter_min = -0.3
        self.y_filter_max = 0.3
        self.z_filter_min = -0.6
        self.z_filter_max = 0.0

    def is_in_filter_box(self, point):
        # Check if the point is in the filter box
        is_in_box = point[0] > self.x_filter_min and \
                    point[0] < self.x_filter_max and \
                    point[1] > self.y_filter_min and \
                    point[1] < self.y_filter_max and \
                    point[2] > self.z_filter_min and \
                    point[2] < self.z_filter_max
        return is_in_box

    def cloud_callback(self, data):
        if not self.time_stamp_offset_set:
            self.time_stamp_offset = self.get_clock().now().nanoseconds - Time.from_msg(data.header.stamp).nanoseconds
            self.time_stamp_offset_set = True
                
        cloud_arr = pc2.read_points_list(data)
        points = np.array(cloud_arr)

        transformed_points = points.copy()
        body_points = points[:, 0:3] @ self.base_from_lidar_rotation.T
        body_points += self.base_from_lidar_translation
        remove_list = []
        transformed_points = transformed_points.tolist()
        for i in range(len(transformed_points)):
            transformed_points[i][4] = int(transformed_points[i][4])
            if self.is_in_filter_box(body_points[i]):
                remove_list.append(i)

        remove_list.sort(reverse=True)

        for id_to_remove in remove_list:
            del transformed_points[id_to_remove]
        
        elevated_cloud = pc2.create_cloud(data.header, data.fields, transformed_points)
        elevated_cloud.header.stamp = Time(nanoseconds=Time.from_msg(elevated_cloud.header.stamp).nanoseconds + self.time_stamp_offset).to_msg()
        elevated_cloud.header.frame_id = self.RAW_CLOUD_FRAME
        elevated_cloud.is_dense = data.is_dense

        self.cloud_pub.publish(elevated_cloud)

        # The native cloud remains the Point-LIO input.  This separate output
        # is the SEA-Nav body-frame contract: x-forward, y-left, z-up.
        base_points = [list(point) for point in transformed_points]
        for point in base_points:
            base_xyz = (np.asarray(point[0:3], dtype=np.float64) @
                        self.base_from_lidar_rotation.T +
                        self.base_from_lidar_translation)
            point[0], point[1], point[2] = base_xyz.tolist()
        cloud_base = pc2.create_cloud(data.header, data.fields, base_points)
        cloud_base.header.stamp = elevated_cloud.header.stamp
        cloud_base.header.frame_id = self.BASE_CLOUD_FRAME
        cloud_base.is_dense = data.is_dense
        self.cloud_base_pub.publish(cloud_base)
            
    def imu_callback(self, data):    
        angular = self.sensor_rotation @ np.array([
            data.angular_velocity.x,
            data.angular_velocity.y,
            data.angular_velocity.z,
        ])
        angular -= np.array([self.ang_bias_x, self.ang_bias_y, self.ang_bias_z])
        angular[0] += self.ang_z2x_proj * angular[2]
        angular[1] += self.ang_z2y_proj * angular[2]

        acceleration = self.sensor_rotation @ np.array([
            data.linear_acceleration.x,
            data.linear_acceleration.y,
            data.linear_acceleration.z,
        ])
        acceleration -= np.array([self.acc_bias_x, self.acc_bias_y, self.acc_bias_z])
        

        transformed_imu = Imu()
        transformed_imu.header.stamp = data.header.stamp
        transformed_imu.header.frame_id = self.RAW_IMU_FRAME
        transformed_imu.orientation = data.orientation
        transformed_imu.angular_velocity.x = angular[0]
        transformed_imu.angular_velocity.y = angular[1]
        transformed_imu.angular_velocity.z = angular[2]
        transformed_imu.linear_acceleration.x = acceleration[0]
        transformed_imu.linear_acceleration.y = acceleration[1]
        transformed_imu.linear_acceleration.z = acceleration[2]
        
        transformed_imu.header.stamp = Time(nanoseconds=Time.from_msg(transformed_imu.header.stamp).nanoseconds + self.time_stamp_offset).to_msg()
        
        self.imu_raw_pub.publish(transformed_imu)
        
        transformed_imu.orientation.x = 0.0
        transformed_imu.orientation.y = 0.0
        transformed_imu.orientation.z = 0.0
        transformed_imu.orientation.w = 1.0
        
        transformed_imu.linear_acceleration.x = 0.0
        transformed_imu.linear_acceleration.y = 0.0
        transformed_imu.linear_acceleration.z = 0.0
        
        self.imu_pub.publish(transformed_imu)

def main(args=None):
    rclpy.init(args=args)

    transform_node = Repuber()

    rclpy.spin(transform_node)

    Repuber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

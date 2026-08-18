from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare('point_lio_unilidar'),
        'config',
        'sea_nav_go2.yaml',
    ])

    sensor_transform = Node(
        package='transform_sensors',
        executable='transform_everything',
        name='sea_nav_sensor_transform',
        output='screen',
        remappings=[
            ('/utlidar/transformed_raw_imu', '/sea_nav/lio/transformed_raw_imu'),
            ('/utlidar/transformed_imu', '/sea_nav/lio/transformed_imu'),
            ('/utlidar/transformed_cloud', '/sea_nav/lio/transformed_cloud'),
        ],
    )

    point_lio = Node(
        package='point_lio_unilidar',
        executable='pointlio_mapping',
        name='sea_nav_point_lio',
        output='screen',
        parameters=[config],
        remappings=[
            ('/cloud_registered', '/sea_nav/lio/cloud_registered'),
            ('/cloud_registered_body', '/sea_nav/lio/cloud_registered_body'),
            ('/aft_mapped_to_init', '/sea_nav/lio/odom'),
            ('/path', '/sea_nav/lio/path'),
        ],
    )

    return LaunchDescription([sensor_transform, point_lio])

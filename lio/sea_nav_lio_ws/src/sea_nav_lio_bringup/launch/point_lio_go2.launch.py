from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    projection_x = LaunchConfiguration('imu_ang_z2x_proj')
    projection_y = LaunchConfiguration('imu_ang_z2y_proj')
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
        parameters=[{
            'imu_ang_z2x_proj': ParameterValue(projection_x, value_type=float),
            'imu_ang_z2y_proj': ParameterValue(projection_y, value_type=float),
        }],
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

    return LaunchDescription([
        DeclareLaunchArgument('imu_ang_z2x_proj', default_value='nan'),
        DeclareLaunchArgument('imu_ang_z2y_proj', default_value='nan'),
        sensor_transform,
        point_lio,
    ])

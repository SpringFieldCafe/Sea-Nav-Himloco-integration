from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.actions import SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    projection_x = LaunchConfiguration('imu_ang_z2x_proj')
    projection_y = LaunchConfiguration('imu_ang_z2y_proj')
    deskew_enabled = LaunchConfiguration('deskew')
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

    odom_adapter = Node(
        package='sea_nav_lio_bringup',
        executable='odom_se2_adapter',
        name='sea_nav_lio_odom_adapter',
        output='screen',
        arguments=[
            '--input-topic', '/sea_nav/lio/odom',
            '--output-topic', '/utlidar/robot_odom',
        ],
    )

    deskew = ExecuteProcess(
        cmd=['/usr/bin/python3', PathJoinSubstitution([
            FindPackageShare('sea_nav_lio_bringup'),
            'deskew_node.py',
        ])],
        output='screen',
        condition=IfCondition(deskew_enabled),
    )

    point_lio_group = GroupAction(
        scoped=True,
        actions=[
            SetRemap(
                src='/sea_nav/lio/transformed_cloud',
                dst='/sea_nav/lio/deskewed_cloud',
                condition=IfCondition(deskew_enabled),
            ),
            point_lio,
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('imu_ang_z2x_proj', default_value='nan'),
        DeclareLaunchArgument('imu_ang_z2y_proj', default_value='nan'),
        DeclareLaunchArgument('deskew', default_value='false'),
        sensor_transform,
        deskew,
        point_lio_group,
        odom_adapter,
    ])

from setuptools import setup

package_name = 'sea_nav_lio_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/point_lio_go2.launch.py']),
        ('share/' + package_name, [
            '../../../../tools/ros2_lidar_deskew/deskew_node.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sea_nav',
    maintainer_email='hyz@example.com',
    description='Read-only Go2 Point-LIO bringup for SEA-Nav odometry.',
    license='GPL-2.0-or-later',
    entry_points={
        'console_scripts': [
            'odom_se2_adapter = sea_nav_lio_bringup.odom_se2_adapter:main',
        ],
    },
)

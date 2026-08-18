from setuptools import setup

package_name = 'sea_nav_lio_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/point_lio_go2.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sea_nav',
    maintainer_email='hyz@example.com',
    description='Read-only Go2 Point-LIO bringup for SEA-Nav odometry.',
    license='GPL-2.0-or-later',
)

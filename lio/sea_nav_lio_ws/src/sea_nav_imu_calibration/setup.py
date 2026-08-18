from setuptools import setup


package_name = 'sea_nav_imu_calibration'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Read-only IMU recorder and offline calibration estimator.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'record_imu = sea_nav_imu_calibration.record_imu:main',
            'estimate_calibration = sea_nav_imu_calibration.estimate_calibration:main',
        ],
    },
)

from setuptools import setup
import os
from glob import glob

package_name = 'nav_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/maps', glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Navigation and Control Package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_planner = nav_control.path_planner:main',
            'rl_follower = nav_control.rl_follower:main',
            'pid_follower = nav_control.pid_follower:main',
            'odom_tf = nav_control.odom_tf:main',
            'scan_converter = nav_control.scan_converter:main',
        ],
    },
)

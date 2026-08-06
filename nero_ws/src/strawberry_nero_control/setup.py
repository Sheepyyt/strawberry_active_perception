"""Package strawberry_nero_control for ROS 2."""

import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'strawberry_nero_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        ('share/' + package_name, ['README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yyt',
    maintainer_email='22313022@zju.edu.cn',
    description='Continuous Placo IK and safe ROS 2 control for a NERO arm.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'nero_control_node = strawberry_nero_control.control_node:main',
            'nero_meshcat_viewer = strawberry_nero_control.meshcat_viewer:main',
            'nero_offline_benchmark = strawberry_nero_control.offline_benchmark:main',
        ],
    },
)

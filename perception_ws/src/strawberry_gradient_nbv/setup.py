"""ROS 2 package metadata for the independent Gradient-NBV implementation."""

import os
from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "strawberry_gradient_nbv"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + PACKAGE_NAME],
        ),
        (
            "share/" + PACKAGE_NAME,
            ["package.xml", "README.md", "NOTICE", "requirements-nbv.txt"],
        ),
        (os.path.join("share", PACKAGE_NAME, "config"), glob("config/*.yaml")),
        (os.path.join("share", PACKAGE_NAME, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yyt",
    maintainer_email="22313022@zju.edu.cn",
    description="MoveIt-free Gradient-NBV core and ROS 2 wrapper.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "gradient_nbv = strawberry_gradient_nbv.ros_node:main",
            "gradient_nbv_fixture = strawberry_gradient_nbv.replay:main",
            "gradient_nbv_ros_replay = strawberry_gradient_nbv.ros_replay:main",
        ],
    },
)

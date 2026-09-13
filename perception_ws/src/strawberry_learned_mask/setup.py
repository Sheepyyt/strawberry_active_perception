"""ROS 2 package metadata for the learned strawberry mask provider."""

import os
from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "strawberry_learned_mask"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + PACKAGE_NAME],
        ),
        ("share/" + PACKAGE_NAME, ["package.xml", "README.md"]),
        (os.path.join("share", PACKAGE_NAME, "config"), glob("config/*.yaml")),
        (os.path.join("share", PACKAGE_NAME, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yyt",
    maintainer_email="22313022@zju.edu.cn",
    description="Audited YOLO11 strawberry instance-mask provider.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "learned_mask_node = strawberry_learned_mask.ros_node:main",
            "learned_mask_offline = strawberry_learned_mask.offline:main",
        ],
    },
)

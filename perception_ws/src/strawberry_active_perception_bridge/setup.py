"""ROS 2 package metadata for NBV preview and one-shot supervision."""

import os
from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "strawberry_active_perception_bridge"


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
    description=(
        "Gradient-NBV to Placo preview and SHA-bound one-shot supervisor."
    ),
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "nbv_ik_preview = "
            "strawberry_active_perception_bridge.preview_node:main",
            "nbv_ik_preview_fixture = "
            "strawberry_active_perception_bridge.simulation_fixture:main",
            "gradient_placo_pipeline_fixture = "
            "strawberry_active_perception_bridge.gradient_pipeline_fixture:main",
            "real_handeye_preview = "
            "strawberry_active_perception_bridge.real_preview_fixture:main",
            "real_nbv_supervisor = "
            "strawberry_active_perception_bridge.real_nbv_supervisor:main",
        ],
    },
)

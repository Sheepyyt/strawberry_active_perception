"""Launch the hash-pinned YOLO11 mask provider only."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("strawberry_learned_mask")
    config_file = LaunchConfiguration("config_file")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=os.path.join(
                    share, "config", "yolo11m_strawberry.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value=(
                    "/home/yyt/strawberry_active_perception/artifacts/models/"
                    "yolo11m_strawberry_best.pt"
                ),
            ),
            Node(
                package="strawberry_learned_mask",
                executable="learned_mask_node",
                name="strawberry_learned_mask",
                output="screen",
                parameters=[
                    config_file,
                    {"checkpoint_path": checkpoint_path},
                ],
            ),
        ]
    )

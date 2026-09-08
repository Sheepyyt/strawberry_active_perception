"""Launch the MoveIt-free ROS 2 Gradient-NBV wrapper."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create one configured NBV action server."""
    share = Path(get_package_share_directory("strawberry_gradient_nbv"))
    config = DeclareLaunchArgument(
        "config_file",
        default_value=str(share / "config" / "nbv.yaml"),
    )
    node = Node(
        package="strawberry_gradient_nbv",
        executable="gradient_nbv",
        name="gradient_nbv",
        output="screen",
        parameters=[LaunchConfiguration("config_file")],
    )
    return LaunchDescription([config, node])

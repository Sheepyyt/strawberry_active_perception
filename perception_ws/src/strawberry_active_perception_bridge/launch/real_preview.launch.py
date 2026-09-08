"""Launch only the report-bound, motion-free real-camera IK bridge."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Start no driver, controller, robot state publisher, or motion client."""
    bridge_share = Path(
        get_package_share_directory("strawberry_active_perception_bridge")
    )
    bridge_config = DeclareLaunchArgument(
        "bridge_config",
        default_value=str(bridge_share / "config" / "real_preview.yaml"),
        description=(
            "Parameters binding the accepted hand-eye report by exact SHA256"
        ),
    )
    bridge = Node(
        package="strawberry_active_perception_bridge",
        executable="nbv_ik_preview",
        name="nbv_ik_preview",
        output="screen",
        parameters=[LaunchConfiguration("bridge_config")],
    )
    return LaunchDescription([bridge_config, bridge])

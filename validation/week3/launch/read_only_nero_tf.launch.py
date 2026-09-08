"""只读 NERO robot_state_publisher；仅消费真实关节反馈并发布 TF。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    description_share = Path(get_package_share_directory("agx_arm_description"))
    urdf_path = (
        description_share
        / "agx_arm_urdf"
        / "nero"
        / "urdf"
        / "nero_description.urdf"
    )
    robot_description = urdf_path.read_text(encoding="utf-8")
    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="handeye_read_only_robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
                remappings=[("joint_states", "/feedback/joint_states")],
            )
        ]
    )


def main() -> int:
    """允许从非 ROS package 的 validation 目录直接安全启动。"""
    service = LaunchService()
    service.include_launch_description(generate_launch_description())
    return service.run()


if __name__ == "__main__":
    raise SystemExit(main())

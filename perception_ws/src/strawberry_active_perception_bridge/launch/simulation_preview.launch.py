"""Launch a CAN-free NERO SolveIK server and the read-only NBV bridge."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    bridge_share = Path(
        get_package_share_directory("strawberry_active_perception_bridge")
    )
    control_share = Path(get_package_share_directory("strawberry_nero_control"))
    description_share = Path(get_package_share_directory("agx_arm_description"))
    urdf_path = (
        description_share
        / "agx_arm_urdf"
        / "nero"
        / "urdf"
        / "nero_description.urdf"
    )
    robot_description = urdf_path.read_text(encoding="utf-8")

    bridge_config = DeclareLaunchArgument(
        "bridge_config",
        default_value=str(bridge_share / "config" / "preview.yaml"),
    )
    control_config = DeclareLaunchArgument(
        "control_config",
        default_value=str(control_share / "config" / "nero_control.yaml"),
    )

    controller = Node(
        package="strawberry_nero_control",
        executable="nero_control_node",
        name="nero_control",
        output="screen",
        parameters=[
            LaunchConfiguration("control_config"),
            {
                "simulation_mode": True,
                "execution_enabled_on_start": False,
                "require_arm_status": False,
            },
        ],
    )
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="nero_robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": ParameterValue(
                    robot_description,
                    value_type=str,
                )
            }
        ],
        remappings=[("/joint_states", "/feedback/joint_states")],
    )
    simulated_mount = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="simulation_camera_mount",
        output="screen",
        arguments=[
            "--x",
            "0.10",
            "--y",
            "0.0",
            "--z",
            "0.0",
            "--qx",
            "0.5",
            "--qy",
            "-0.5",
            "--qz",
            "0.5",
            "--qw",
            "-0.5",
            "--frame-id",
            "link7",
            "--child-frame-id",
            "camera_sim_optical_frame",
        ],
    )
    bridge = Node(
        package="strawberry_active_perception_bridge",
        executable="nbv_ik_preview",
        name="nbv_ik_preview",
        output="screen",
        parameters=[LaunchConfiguration("bridge_config")],
    )
    gradient_nbv = Node(
        package="strawberry_gradient_nbv",
        executable="gradient_nbv",
        name="gradient_nbv",
        output="screen",
    )
    # This launch intentionally has no joint-command publisher and never starts
    # the vendor driver.  Keep an explicit machine-checkable safety assertion in
    # the launch description so every preview run records the closed gate.
    safety_marker = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="preview_no_motion_marker",
        output="log",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
            "--frame-id", "active_perception_preview_execution_false",
            "--child-frame-id", "active_perception_preview_motion_commands_0",
        ],
    )
    return LaunchDescription(
        [
            bridge_config,
            control_config,
            controller,
            state_publisher,
            simulated_mount,
            safety_marker,
            gradient_nbv,
            bridge,
        ]
    )

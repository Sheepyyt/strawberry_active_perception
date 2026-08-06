"""Start a real NERO in read-only mode plus the disabled Placo controller."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def _launch_path(package_name, filename):
    share = Path(get_package_share_directory(package_name))
    return PythonLaunchDescriptionSource(str(share / 'launch' / filename))


def generate_launch_description():
    """Build the initially read-only real-arm launch description."""
    package_share = Path(
        get_package_share_directory('strawberry_nero_control')
    )
    default_config = str(package_share / 'config' / 'nero_control.yaml')

    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='NERO Placo controller parameter file.',
    )
    can_port_arg = DeclareLaunchArgument(
        'can_port',
        default_value='can0',
        description='SocketCAN interface connected to this NERO.',
    )
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='false',
        choices=['true', 'false'],
        description='Open read-only RViz visualization of measured joints.',
    )

    driver = IncludeLaunchDescription(
        _launch_path('agx_arm_ctrl', 'start_single_agx_arm.launch.py'),
        launch_arguments={
            'can_port': LaunchConfiguration('can_port'),
            'arm_type': 'nero',
            'effector_type': 'none',
            'auto_enable': 'false',
            'control_enabled': 'false',
            'fast_mode': 'false',
            'speed_percent': '10',
        }.items(),
    )

    controller = Node(
        package='strawberry_nero_control',
        executable='nero_control_node',
        name='nero_control',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'simulation_mode': False,
                'execution_enabled_on_start': False,
                'require_arm_status': True,
            },
        ],
    )

    # RViz follows feedback only. The joint slider publisher is never started,
    # so the Placo controller remains the sole source of motion commands.
    rviz = IncludeLaunchDescription(
        _launch_path('agx_arm_description', 'display.launch.py'),
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
        launch_arguments={
            'arm_type': 'nero',
            'effector_type': 'none',
            'follow': 'true',
            'control': 'false',
            'gui': 'false',
            'feedback_topic': '/feedback/joint_states',
        }.items(),
    )

    return LaunchDescription(
        [config_arg, can_port_arg, launch_rviz_arg, driver, controller, rviz]
    )

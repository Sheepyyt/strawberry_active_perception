"""Start Placo control and MeshCat without opening a CAN connection."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Build a CAN-free controller and visualization launch description."""
    package_share = Path(
        get_package_share_directory('strawberry_nero_control')
    )
    default_config = str(package_share / 'config' / 'nero_control.yaml')

    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='NERO Placo controller parameter file.',
    )
    meshcat_port_arg = DeclareLaunchArgument(
        'meshcat_port',
        default_value='7000',
        description='TCP port used by the MeshCat viewer.',
    )

    common_parameters = [
        LaunchConfiguration('config_file'),
        {
            'simulation_mode': True,
            'execution_enabled_on_start': True,
            'require_arm_status': False,
        },
    ]

    controller = Node(
        package='strawberry_nero_control',
        executable='nero_control_node',
        name='nero_control',
        output='screen',
        parameters=common_parameters,
    )
    viewer = Node(
        package='strawberry_nero_control',
        executable='nero_meshcat_viewer',
        name='nero_meshcat_viewer',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'simulation_mode': True,
                'meshcat_port': ParameterValue(
                    LaunchConfiguration('meshcat_port'), value_type=int
                ),
            },
        ],
    )

    # No agx_arm_ctrl include is allowed here: sim must never touch CAN.
    return LaunchDescription(
        [config_arg, meshcat_port_arg, controller, viewer]
    )

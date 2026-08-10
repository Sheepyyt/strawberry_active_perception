"""Start a real NERO with guarded, explicitly selected motor enable state."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _launch_path(package_name, filename):
    share = Path(get_package_share_directory(package_name))
    return PythonLaunchDescriptionSource(str(share / 'launch' / filename))


def _validate_supervised_options(context):
    allow_recovery = (
        LaunchConfiguration('allow_limit_recovery_execution').perform(context)
        == 'true'
    )
    speed_percent = int(LaunchConfiguration('speed_percent').perform(context))
    first_motion = (
        LaunchConfiguration('first_motion_test_mode').perform(context)
        == 'true'
    )
    precision_test = (
        LaunchConfiguration('precision_test_mode').perform(context)
        == 'true'
    )
    startup_enable = (
        LaunchConfiguration('startup_enable').perform(context) == 'true'
    )
    if first_motion and precision_test:
        raise RuntimeError(
            'first_motion_test_mode and precision_test_mode are mutually exclusive'
        )
    supervised = (
        allow_recovery or first_motion or precision_test or startup_enable
    )
    if supervised and speed_percent != 10:
        raise RuntimeError(
            'supervised recovery/test/startup-enable modes require '
            'speed_percent=10'
        )
    if startup_enable and not (
        allow_recovery or first_motion or precision_test
    ):
        raise RuntimeError(
            'startup_enable=true is only allowed in an explicit supervised '
            'recovery or test mode'
        )
    return []


def generate_launch_description():
    """Build the guarded real-arm launch description."""
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
    speed_percent_arg = DeclareLaunchArgument(
        'speed_percent',
        default_value='0',
        description=(
            'Driver speed setting. Keep 0 during read-only checks; pass 10 '
            'only after the hardware checklist is complete.'
        ),
    )
    firmware_arg = DeclareLaunchArgument(
        'nero_fw_version',
        default_value='v111',
        description=(
            'Verified NERO controller firmware. An explicit value avoids '
            'firmware auto-detection while the arm is disabled.'
        ),
    )
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='false',
        choices=['true', 'false'],
        description='Open read-only RViz visualization of measured joints.',
    )
    recovery_execution_arg = DeclareLaunchArgument(
        'allow_limit_recovery_execution',
        default_value='false',
        choices=['true', 'false'],
        description=(
            'Unlock the dedicated inward-only joint-limit recovery action. '
            'Keep false for read-only checks and preview.'
        ),
    )
    startup_enable_arg = DeclareLaunchArgument(
        'startup_enable',
        default_value='false',
        choices=['true', 'false'],
        description=(
            'Explicitly enable NERO motors during driver startup. This is '
            'only for a supervised cold start that produces no feedback '
            'while disabled; it does not open either motion-control gate.'
        ),
    )
    first_motion_test_arg = DeclareLaunchArgument(
        'first_motion_test_mode',
        default_value='false',
        choices=['true', 'false'],
        description=(
            'Tighten tracking and final-error gates for the supervised first '
            '15 mm Placo test.'
        ),
    )
    precision_test_arg = DeclareLaunchArgument(
        'precision_test_mode',
        default_value='false',
        choices=['true', 'false'],
        description=(
            'Tighten tracking and IK gates for supervised small repeatability '
            'tests without the one-shot 15 mm lock.'
        ),
    )

    driver = IncludeLaunchDescription(
        _launch_path('agx_arm_ctrl', 'start_single_agx_arm.launch.py'),
        launch_arguments={
            'can_port': LaunchConfiguration('can_port'),
            'arm_type': 'nero',
            'effector_type': 'none',
            'auto_enable': LaunchConfiguration('startup_enable'),
            'control_enabled': 'false',
            'fast_mode': 'false',
            'speed_percent': LaunchConfiguration('speed_percent'),
            'fw_version': LaunchConfiguration('nero_fw_version'),
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
                'allow_limit_recovery_execution': ParameterValue(
                    LaunchConfiguration('allow_limit_recovery_execution'),
                    value_type=bool,
                ),
                'verified_driver_speed_percent': ParameterValue(
                    LaunchConfiguration('speed_percent'),
                    value_type=int,
                ),
                'first_motion_test_mode': ParameterValue(
                    LaunchConfiguration('first_motion_test_mode'),
                    value_type=bool,
                ),
                'precision_test_mode': ParameterValue(
                    LaunchConfiguration('precision_test_mode'),
                    value_type=bool,
                ),
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
        [
            config_arg,
            can_port_arg,
            speed_percent_arg,
            firmware_arg,
            launch_rviz_arg,
            recovery_execution_arg,
            startup_enable_arg,
            first_motion_test_arg,
            precision_test_arg,
            OpaqueFunction(function=_validate_supervised_options),
            driver,
            controller,
            rviz,
        ]
    )

"""Start Gemini 2 XL in the minimum aligned RGB-D mode plus capture adapter."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    adapter_share = get_package_share_directory('strawberry_observation')
    orbbec_share = get_package_share_directory('orbbec_camera')

    serial_number = LaunchConfiguration('serial_number')
    config_file = LaunchConfiguration('config_file')

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([orbbec_share, 'launch', 'gemini2XL.launch.py'])
        ),
        launch_arguments={
            'camera_name': 'camera',
            'serial_number': serial_number,
            'color_width': '640',
            'color_height': '400',
            'color_fps': '10',
            'color_format': 'MJPG',
            'depth_width': '640',
            'depth_height': '400',
            'depth_fps': '10',
            'depth_format': 'Y16',
            'enable_color': 'true',
            'enable_depth': 'true',
            'depth_registration': 'true',
            'align_mode': 'HW',
            'enable_frame_sync': 'true',
            'enable_depth_scale': 'true',
            'enable_left_ir': 'false',
            'enable_right_ir': 'false',
            'enable_accel': 'false',
            'enable_gyro': 'false',
            'enable_sync_output_accel_gyro': 'false',
            'enable_point_cloud': 'false',
            'enable_colored_point_cloud': 'false',
            'retry_on_usb3_detection_failure': 'true',
        }.items(),
    )

    # The vendor Gemini2XL launch does not declare/pass time_domain. A scoped
    # global parameter is therefore required. Colour rectification deliberately
    # stays in strawberry_observation: the vendor undistortion path both fails
    # to activate reliably and clears the coefficients before remapping.
    camera_group = GroupAction(
        scoped=True,
        actions=[
            SetParameter(name='time_domain', value='global'),
            camera,
        ],
    )

    adapter = Node(
        package='strawberry_observation',
        executable='strawberry_observation_node',
        name='strawberry_observation',
        output='screen',
        parameters=[config_file],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'serial_number',
                default_value='AYML241003A',
                description='Exact Gemini 2 XL serial number',
            ),
            DeclareLaunchArgument(
                'config_file',
                default_value=os.path.join(
                    adapter_share, 'config', 'gemini2xl_observation.yaml'
                ),
                description='Observation adapter parameter YAML',
            ),
            camera_group,
            adapter,
        ]
    )

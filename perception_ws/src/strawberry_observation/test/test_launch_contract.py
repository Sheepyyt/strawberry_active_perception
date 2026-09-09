"""Guard the vendor-launch workaround and minimum camera bandwidth surface."""

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_ros.actions import SetParameter


def _load_launch_module():
    path = Path(__file__).parents[1] / 'launch' / 'gemini2xl_observation.launch.py'
    spec = importlib.util.spec_from_file_location('gemini2xl_observation_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_time_domain_is_scoped_parameter_before_vendor_include(monkeypatch, tmp_path):
    module = _load_launch_module()

    def fake_share(package_name):
        share = tmp_path / package_name
        (share / 'launch').mkdir(parents=True, exist_ok=True)
        if package_name == 'orbbec_camera':
            (share / 'launch' / 'gemini2XL.launch.py').write_text(
                'from launch import LaunchDescription\n'
                'def generate_launch_description(): return LaunchDescription([])\n',
                encoding='utf-8',
            )
        return str(share)

    monkeypatch.setattr(module, 'get_package_share_directory', fake_share)
    description = module.generate_launch_description()
    group = next(entity for entity in description.entities if isinstance(entity, GroupAction))
    group_entities = group.get_sub_entities()
    include_index = next(
        index for index, entity in enumerate(group_entities)
        if isinstance(entity, IncludeLaunchDescription)
    )
    parameters = [
        entity for entity in group_entities[:include_index]
        if isinstance(entity, SetParameter)
    ]
    assert len(parameters) == 1
    assert all(index < include_index for index, entity in enumerate(group_entities)
               if isinstance(entity, SetParameter))

    context = LaunchContext()
    for parameter in parameters:
        parameter.execute(context)
    assert ('time_domain', 'global') in context.launch_configurations['global_params']
    assert not any(
        name == 'enable_color_undistortion'
        for name, _ in context.launch_configurations['global_params']
    )

    source = (Path(__file__).parents[1] / 'launch' /
              'gemini2xl_observation.launch.py').read_text(encoding='utf-8')
    assert 'scoped=True' in source
    assert "SetParameter(name='enable_color_undistortion'" not in source


def test_minimum_camera_mode_is_explicit_in_wrapper_source():
    source = (
        Path(__file__).parents[1] / 'launch' / 'gemini2xl_observation.launch.py'
    ).read_text(encoding='utf-8')
    for setting in (
        "'depth_registration': 'true'",
        "'align_mode': 'HW'",
        "'enable_frame_sync': 'true'",
        "'enable_left_ir': 'false'",
        "'enable_right_ir': 'false'",
        "'enable_sync_output_accel_gyro': 'false'",
        "'enable_point_cloud': 'false'",
        "'enable_colored_point_cloud': 'false'",
    ):
        assert setting in source


def test_adapter_config_declares_real_source_type():
    config = (
        Path(__file__).parents[1] / 'config' / 'gemini2xl_observation.yaml'
    ).read_text(encoding='utf-8')
    assert 'source_type: 1' in config
    assert 'input_color_topic: /camera/color/image_raw' in config
    assert 'input_color_camera_info_topic: /camera/color/camera_info' in config
    assert 'input_depth_camera_info_topic: /camera/depth/camera_info' in config
    assert 'mask_dilation_kernel_size: 5' in config


def test_adapter_source_uses_four_way_sync_and_both_camera_infos():
    source = (
        Path(__file__).parents[1] / 'src' / 'strawberry_observation_node.cpp'
    ).read_text(encoding='utf-8')
    assert 'Image, Image, CameraInfo, CameraInfo' in source
    assert 'color_camera_info_subscriber_' in source
    assert 'depth_camera_info_subscriber_' in source
    assert '*sample.color_camera_info' in source
    assert '*sample.depth_camera_info' in source

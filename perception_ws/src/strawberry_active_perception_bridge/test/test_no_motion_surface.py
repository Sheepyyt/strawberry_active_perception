"""Guard against accidentally broadening preview into robot execution."""

from pathlib import Path


def test_preview_node_has_no_motion_action_or_can_publisher() -> None:
    package = Path(__file__).parents[1] / "strawberry_active_perception_bridge"
    source = (package / "preview_node.py").read_text(encoding="utf-8")
    forbidden_symbols = (
        "MoveToPose",
        "strawberry_nero_interfaces.action",
        "agx_arm_ctrl",
    )
    assert all(symbol not in source for symbol in forbidden_symbols)
    assert "from sensor_msgs.msg import JointState" in source
    assert "self.create_subscription(\n            JointState" in source
    assert "self.create_publisher(\n            JointState" not in source


def test_real_fixture_has_observers_but_no_motion_surface() -> None:
    package = Path(__file__).parents[1] / "strawberry_active_perception_bridge"
    source = (package / "real_preview_fixture.py").read_text(encoding="utf-8")
    forbidden_symbols = (
        "MoveToPose",
        "ActionClient",
        "strawberry_nero_interfaces.action",
        "strawberry_nero_interfaces.srv",
        "agx_arm_ctrl",
    )
    assert all(symbol not in source for symbol in forbidden_symbols)
    assert "node.create_subscription(\n        JointState" in source
    assert "node.create_publisher(\n        JointState" not in source


def test_fixtures_only_observe_the_command_topic() -> None:
    package = Path(__file__).parents[1] / "strawberry_active_perception_bridge"
    source = (package / "gradient_pipeline_fixture.py").read_text(
        encoding="utf-8"
    )
    assert '"/control/move_j"' in source
    assert "strawberry_nero_interfaces.action" not in source
    assert "MoveToPose" not in source
    # The only JointState ROS entity is the independent command subscriber.
    assert "create_publisher(\n        JointState" not in source
    assert "create_subscription(\n        JointState" in source


def test_simulation_launch_keeps_execution_gate_closed() -> None:
    source = (
        Path(__file__).parents[1] / "launch" / "simulation_preview.launch.py"
    ).read_text(encoding="utf-8")
    assert '"execution_enabled_on_start": False' in source
    assert "active_perception_preview_motion_commands_0" in source
    assert "agx_arm_ctrl" not in source


def test_real_launch_starts_only_the_report_bound_bridge() -> None:
    root = Path(__file__).parents[1]
    launch_source = (root / "launch" / "real_preview.launch.py").read_text(
        encoding="utf-8"
    )
    assert launch_source.count("Node(") == 1
    assert 'executable="nbv_ik_preview"' in launch_source
    forbidden = (
        "strawberry_nero_control",
        "robot_state_publisher",
        "static_transform_publisher",
        "agx_arm_ctrl",
        "gradient_nbv",
    )
    assert all(symbol not in launch_source for symbol in forbidden)


def test_real_config_is_hash_bound_and_has_no_copied_transform() -> None:
    root = Path(__file__).parents[1]
    source = (root / "config" / "real_preview.yaml").read_text(encoding="utf-8")
    assert "calibration_source: validated_report" in source
    assert "expected_camera_frame: camera_color_optical_frame" in source
    assert "base_frame: base_link" in source
    assert (
        "handeye_report_sha256: "
        "31eb93b2b80663b895eac564afc8f633b4310a6b7c5e519340d97d163f22825f"
        in source
    )
    assert "link7_camera_transform" not in source

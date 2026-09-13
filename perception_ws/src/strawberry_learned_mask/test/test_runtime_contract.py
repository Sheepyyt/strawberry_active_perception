"""Static tests keep the learned-mask node isolated from robot control."""

import ast
from pathlib import Path

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
NODE = PACKAGE / "strawberry_learned_mask" / "ros_node.py"
CONFIG = PACKAGE / "config" / "yolo11m_strawberry.yaml"
OVERLAY = PACKAGE / "config" / "real_nbv_use_learned_mask.yaml"


def test_node_has_no_service_action_or_robot_control_surface():
    source = NODE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "create_client" not in attributes
    assert "create_service" not in attributes
    assert "ActionClient" not in source
    assert "MoveToPose" not in source
    assert "/control/" not in source
    assert "/feedback/" not in source


def test_config_is_hash_pinned_single_strawberry_high_confidence():
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    params = payload["strawberry_learned_mask"]["ros__parameters"]
    assert params["checkpoint_sha256"] == (
        "7bea8d97b68c8081f1949538ec8a6ef14324c1f9ab9ae1b75ddefd2889c49357"
    )
    assert params["allowed_labels"] == ["strawberry"]
    assert params["confidence_threshold"] >= 0.70
    assert params["instance_policy"] == "highest_confidence"
    assert params["minimum_mask_pixels"] >= 200
    assert params["minimum_valid_depth_pixels"] >= 100
    assert params["device"] == "cpu"
    assert not str(params["checkpoint_path"]).startswith("/")
    assert not str(params["audit_directory"]).startswith("/")


def test_nbv_overlay_changes_only_observation_input_topic():
    payload = yaml.safe_load(OVERLAY.read_text(encoding="utf-8"))
    params = payload["real_nbv_supervisor"]["ros__parameters"]
    assert params == {
        "raw_observation_topic": "/strawberry/perception/learned_observation",
        "upstream_mask_required": False,
    }

"""Accepted/rejected frames must retain truthful logs and zero-mask decisions."""

import json
from types import SimpleNamespace

import numpy as np
from rclpy.logging import get_logger

from strawberry_learned_mask.provider import InstancePrediction
from strawberry_learned_mask.ros_node import LearnedMaskNode


def test_depth_rejection_between_accepted_frames_has_no_logging_failure(tmp_path, capfd):
    """Exercise rclpy's real callsite cache without constructing a ROS node."""
    published = []
    header = SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0))
    color = SimpleNamespace(header=header, encoding="rgb8", height=20,
                            width=20, step=60, data=bytes(1200))
    depth = SimpleNamespace(encoding="32FC1", height=20, width=20, step=80,
                            is_bigendian=0, data=b"")
    message = SimpleNamespace(header=header, color=color, depth=depth,
                              target_mask=SimpleNamespace(), scene_id="logging_test",
                              observation_id="", source_name="fixture")
    prediction = InstancePrediction(0, "strawberry", 0.92, np.ones((20, 20)))
    logger = get_logger("learned_mask_logging_regression")
    adapter = SimpleNamespace(
        backend=SimpleNamespace(predict=lambda *args: ([prediction], {})),
        confidence_threshold=.7, mask_threshold=.5,
        allowed_labels=("strawberry",), instance_policy="highest_confidence",
        minimum_mask_pixels=200, minimum_valid_depth_pixels=100,
        checkpoint_sha256="a" * 64, audit_directory=tmp_path,
        publisher=SimpleNamespace(publish=published.append), get_logger=lambda: logger,
    )
    for index, valid_pixels in enumerate((400, 96, 400)):
        values = np.full((20, 20), np.nan, dtype=np.float32)
        values.flat[:valid_pixels] = .34
        depth.data = values.tobytes()
        message.observation_id = f"frame_{index}"
        LearnedMaskNode._on_observation(adapter, message)

    assert len(published) == 3
    published_pixels = [
        sum(value != 0 for value in item.target_mask.data) for item in published
    ]
    assert published_pixels == [400, 0, 400]
    audit = json.loads((tmp_path / "logging_test_frame_1.json").read_text())
    assert audit["accepted_for_nbv"] is False
    assert audit["accepted_instances"][0]["confidence"] == .92
    assert audit["rejection_reasons"] == [
        "valid_depth_pixels=96 below minimum_valid_depth_pixels=100"
    ]
    logs = capfd.readouterr().err
    assert "Logger severity cannot be changed" not in logs
    assert "processing failed" not in logs
    assert "valid_depth_pixels=96" in logs

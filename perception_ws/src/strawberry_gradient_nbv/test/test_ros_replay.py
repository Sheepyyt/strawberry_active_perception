"""Semantic equivalence tests for fixture-to-ROS Topic replay."""

import copy

import numpy as np
from rclpy.serialization import deserialize_message, serialize_message

from strawberry_gradient_nbv.fixtures import (
    make_multiview_fixture,
    save_fixture_npz,
)
from strawberry_gradient_nbv.image_codec import (
    camera_matrix,
    decode_depth_32fc1_m,
    decode_mono8,
    decode_rgb8,
    pose_matrix,
)
from strawberry_gradient_nbv.replay import NPZReplay
from strawberry_gradient_nbv.ros_replay import (
    canonical_observation_from_fixture,
    fixture_to_observation,
)
from strawberry_perception_interfaces.msg import Observation


def test_ros_replay_preserves_every_canonical_numerical_field() -> None:
    fixture = make_multiview_fixture(96, 64)
    item = next(fixture.observations())
    message = fixture_to_observation(item)
    assert message.scene_id == item.scene_id
    assert message.observation_id == item.observation_id
    assert message.source_type == Observation.SOURCE_REPLAY
    assert message.header == message.depth.header
    assert message.header == message.color.header
    assert message.header == message.target_mask.header
    assert message.header == message.camera_info.header
    assert message.camera_pose.header.stamp == message.header.stamp
    assert message.camera_pose.header.frame_id == fixture.config["world_frame"]
    assert message.pose_valid
    np.testing.assert_array_equal(decode_rgb8(message.color), item.color)
    np.testing.assert_array_equal(decode_mono8(message.target_mask), item.mask)
    np.testing.assert_array_equal(decode_depth_32fc1_m(message.depth), item.depth)
    np.testing.assert_array_equal(camera_matrix(message.camera_info), item.K)
    np.testing.assert_allclose(pose_matrix(message.camera_pose), item.pose, atol=1e-12)
    assert message.valid_depth_fraction == np.float32(
        np.count_nonzero(np.isfinite(item.depth)) / item.depth.size
    )


def test_ros_replay_uses_xyzw_for_nonidentity_pose() -> None:
    item = list(make_multiview_fixture(64, 48).observations())[1]
    message = fixture_to_observation(item)
    quaternion = message.camera_pose.pose.orientation
    assert quaternion.w != 0.0
    np.testing.assert_allclose(pose_matrix(message.camera_pose), item.pose, atol=1e-12)


def test_four_source_kinds_are_wire_equivalent_after_metadata(tmp_path) -> None:
    """All declared ingress kinds share exactly one downstream wire contract.

    The REAL record deliberately starts at the post-C++-adapter boundary.  It
    does not claim to open a Gemini device; adapter normalization has separate
    C++ tests in strawberry_observation.
    """
    fixture = make_multiview_fixture(96, 64)
    direct_item = list(fixture.observations())[1]
    fixture_path = save_fixture_npz(fixture, tmp_path / "four_sources.npz")
    loaded_item = list(NPZReplay(fixture_path))[1]

    messages = {
        "real_wire_simulation": canonical_observation_from_fixture(
            direct_item,
            source_type=Observation.SOURCE_REAL,
            source_name="gemini_adapter_post_normalization_fixture",
        ),
        "offline_npz": canonical_observation_from_fixture(
            loaded_item,
            source_type=Observation.SOURCE_OFFLINE,
            source_name="offline_npz_fixture",
        ),
        "synthetic_direct": canonical_observation_from_fixture(
            direct_item,
            source_type=Observation.SOURCE_SYNTHETIC,
            source_name="deterministic_synthetic_fixture",
        ),
        "ros_replay": fixture_to_observation(
            loaded_item,
            source_name="npz_ros_topic_replay_fixture",
        ),
    }
    assert {message.source_type for message in messages.values()} == {
        Observation.SOURCE_REAL,
        Observation.SOURCE_OFFLINE,
        Observation.SOURCE_SYNTHETIC,
        Observation.SOURCE_REPLAY,
    }

    normalized_roundtrips = []
    for message in messages.values():
        normalized = copy.deepcopy(message)
        normalized.source_type = Observation.SOURCE_UNKNOWN
        normalized.source_name = "source-metadata-excluded"
        normalized_roundtrips.append(
            deserialize_message(serialize_message(normalized), Observation)
        )

        assert message.color.encoding == "rgb8"
        assert message.depth.encoding == "32FC1"
        assert message.target_mask.encoding == "mono8"
        np.testing.assert_array_equal(decode_rgb8(message.color), direct_item.color)
        np.testing.assert_array_equal(
            decode_depth_32fc1_m(message.depth), direct_item.depth
        )
        np.testing.assert_array_equal(
            decode_mono8(message.target_mask), direct_item.mask
        )
        np.testing.assert_array_equal(
            camera_matrix(message.camera_info), direct_item.K
        )
        np.testing.assert_allclose(
            pose_matrix(message.camera_pose), direct_item.pose, atol=1e-12
        )

    assert all(
        message == normalized_roundtrips[0]
        for message in normalized_roundtrips[1:]
    )


def test_wire_factory_rejects_unknown_source_kind() -> None:
    item = next(make_multiview_fixture(64, 48).observations())
    with np.testing.assert_raises_regex(ValueError, "source_type"):
        canonical_observation_from_fixture(
            item,
            source_type=Observation.SOURCE_UNKNOWN,
            source_name="invalid",
        )

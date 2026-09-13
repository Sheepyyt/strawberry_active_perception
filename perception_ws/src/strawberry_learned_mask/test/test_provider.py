"""Pure tests for deterministic learned-mask selection and image decoding."""

from types import SimpleNamespace

import numpy as np
import pytest

from strawberry_learned_mask.provider import (
    InstancePrediction,
    choose_instance_mask,
    decode_depth_32fc1,
    decode_rgb8,
)


def _prediction(label: str, confidence: float, pixels: slice) -> InstancePrediction:
    mask = np.zeros((4, 5), dtype=np.float32)
    mask[pixels, 1:4] = 0.9
    return InstancePrediction(0, label, confidence, mask)


def test_highest_confidence_keeps_one_allowed_strawberry():
    predictions = [
        _prediction("leaf", 0.99, slice(0, 1)),
        _prediction("strawberry", 0.74, slice(0, 2)),
        _prediction("strawberry", 0.93, slice(2, 4)),
    ]
    result = choose_instance_mask(
        predictions,
        (4, 5),
        confidence_threshold=0.70,
        policy="highest_confidence",
    )
    assert len(result.accepted) == 1
    assert result.accepted[0]["confidence"] == pytest.approx(0.93)
    assert np.count_nonzero(result.mask) == 6
    assert {item["reason"] for item in result.rejected} == {"class_not_allowed"}


def test_confidence_filter_and_union_policy_are_deterministic():
    predictions = [
        _prediction("strawberry", 0.69, slice(0, 1)),
        _prediction("Strawberry", 0.80, slice(0, 2)),
        _prediction("strawberry", 0.75, slice(2, 4)),
    ]
    result = choose_instance_mask(
        predictions,
        (4, 5),
        confidence_threshold=0.70,
        policy="union",
    )
    assert len(result.accepted) == 2
    assert np.count_nonzero(result.mask) == 12
    assert result.rejected[0]["reason"] == "confidence_below_threshold"


def test_bad_grid_and_invalid_configuration_are_rejected():
    bad = InstancePrediction(0, "strawberry", 0.9, np.ones((2, 2)))
    result = choose_instance_mask([bad], (4, 5))
    assert not np.any(result.mask)
    assert result.rejected[0]["reason"] == "grid_mismatch"
    with pytest.raises(ValueError, match="policy"):
        choose_instance_mask([], (4, 5), policy="unknown")
    with pytest.raises(ValueError, match="confidence"):
        choose_instance_mask([], (4, 5), confidence_threshold=1.1)


def test_rgb8_decoder_handles_row_padding():
    rows = np.array(
        [
            [1, 2, 3, 4, 5, 6, 99, 99],
            [7, 8, 9, 10, 11, 12, 99, 99],
        ],
        dtype=np.uint8,
    )
    image = SimpleNamespace(
        encoding="rgb8", height=2, width=2, step=8, data=rows.tobytes()
    )
    decoded = decode_rgb8(image)
    np.testing.assert_array_equal(
        decoded,
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]),
    )


@pytest.mark.parametrize("bigendian", [0, b"\x00", 1, b"\x01"])
def test_depth_decoder_handles_padding_and_ros_uint8_bytes(bigendian):
    dtype = ">f4" if bigendian in (1, b"\x01") else "<f4"
    rows = np.array([[1.0, 2.0, 99.0], [3.0, np.nan, 99.0]], dtype=dtype)
    image = SimpleNamespace(
        encoding="32FC1",
        height=2,
        width=2,
        step=12,
        data=rows.tobytes(),
        is_bigendian=bigendian,
    )
    decoded = decode_depth_32fc1(image)
    np.testing.assert_allclose(decoded, [[1.0, 2.0], [3.0, np.nan]], equal_nan=True)

"""Tests for the zero-cv_bridge canonical Observation decoder."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from strawberry_gradient_nbv.image_codec import (
    ObservationDecodeError,
    camera_matrix,
    decode_depth_32fc1_m,
    decode_mono8,
    decode_rgb8,
    pose_matrix,
)


def _image(encoding, width, height, step, data, bigendian=False):
    return SimpleNamespace(
        encoding=encoding,
        width=width,
        height=height,
        step=step,
        data=data,
        is_bigendian=bigendian,
    )


def test_rgb_and_mask_decode_padded_rows() -> None:
    rgb = bytes((1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12, 99, 99))
    result = decode_rgb8(_image("rgb8", 2, 2, 8, rgb))
    assert result.shape == (2, 2, 3)
    assert result.tolist() == [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]

    mask = decode_mono8(_image("mono8", 2, 2, 3, bytes((0, 255, 7, 255, 0, 7))))
    assert mask.tolist() == [[0, 255], [255, 0]]
    with pytest.raises(ObservationDecodeError, match="0 or 255"):
        decode_mono8(_image("mono8", 2, 1, 2, bytes((0, 1))))


def test_depth_decodes_big_endian_and_normalizes_invalid() -> None:
    payload = b"".join(struct.pack(">f", value) for value in (1.25, 0.0, -1.0, float("inf")))
    depth = decode_depth_32fc1_m(_image("32FC1", 2, 2, 8, payload, True))
    assert depth.dtype == np.float32
    assert depth[0, 0] == pytest.approx(1.25)
    assert np.isnan(depth[0, 1:]).all()
    assert np.isnan(depth[1]).all()


def test_invalid_buffer_intrinsics_and_pose_are_rejected() -> None:
    with pytest.raises(ObservationDecodeError, match="buffer"):
        decode_rgb8(_image("rgb8", 2, 2, 6, bytes(3)))
    info = SimpleNamespace(
        k=[500.0, 0.0, 1.0, 0.0, 500.0, 1.0, 0.0, 0.0, 1.0],
        d=[0.0] * 5,
    )
    np.testing.assert_allclose(camera_matrix(info), np.asarray(info.k).reshape(3, 3))
    info.d[0] = -0.2
    with pytest.raises(ObservationDecodeError, match="must be zero"):
        camera_matrix(info)
    info.d[0] = 0.0
    info.k[0] = 0.0
    with pytest.raises(ObservationDecodeError, match="positive"):
        camera_matrix(info)

    pose = SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )
    np.testing.assert_allclose(pose_matrix(pose), np.array(
        [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float
    ))
    pose.pose.orientation.w = 0.0
    with pytest.raises(ObservationDecodeError, match="zero norm"):
        pose_matrix(pose)

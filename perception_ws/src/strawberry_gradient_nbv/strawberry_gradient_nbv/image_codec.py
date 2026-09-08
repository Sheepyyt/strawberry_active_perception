"""Strict ROS image decoding without cv_bridge.

The functions accept message-like objects, which keeps their unit tests usable
without a running ROS graph.  Every returned array owns native-endian memory so
PyTorch never retains a view into a mutable ROS loaned buffer.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ObservationDecodeError(ValueError):
    """A canonical Observation violated its public wire contract."""


def _shape_and_buffer(message: Any, bytes_per_pixel: int) -> tuple[int, int, memoryview]:
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    if height <= 0 or width <= 0:
        raise ObservationDecodeError("image dimensions must be positive")
    packed_step = width * bytes_per_pixel
    if step < packed_step:
        raise ObservationDecodeError(
            f"image step {step} is smaller than packed row size {packed_step}"
        )
    data = memoryview(message.data)
    required = step * height
    if data.nbytes < required:
        raise ObservationDecodeError(
            f"image buffer has {data.nbytes} bytes, expected at least {required}"
        )
    return height, width, data


def decode_rgb8(message: Any) -> np.ndarray:
    """Decode one canonical ``rgb8`` image, including padded rows."""
    if str(message.encoding).lower() != "rgb8":
        raise ObservationDecodeError(
            f"color encoding must be rgb8, got {message.encoding!r}"
        )
    height, width, data = _shape_and_buffer(message, 3)
    view = np.ndarray(
        shape=(height, width, 3),
        dtype=np.uint8,
        buffer=data,
        strides=(int(message.step), 3, 1),
    )
    return np.array(view, dtype=np.uint8, copy=True, order="C")


def decode_mono8(message: Any) -> np.ndarray:
    """Decode and validate a canonical binary target mask."""
    if str(message.encoding).lower() != "mono8":
        raise ObservationDecodeError(
            f"target mask encoding must be mono8, got {message.encoding!r}"
        )
    height, width, data = _shape_and_buffer(message, 1)
    view = np.ndarray(
        shape=(height, width),
        dtype=np.uint8,
        buffer=data,
        strides=(int(message.step), 1),
    )
    result = np.array(view, dtype=np.uint8, copy=True, order="C")
    values = np.unique(result)
    if not np.all(np.isin(values, np.array((0, 255), dtype=np.uint8))):
        raise ObservationDecodeError("target mask values must be exactly 0 or 255")
    return result


def decode_depth_32fc1_m(message: Any) -> np.ndarray:
    """Decode native/big-endian ``32FC1`` and normalize invalid values to NaN."""
    if str(message.encoding).upper() != "32FC1":
        raise ObservationDecodeError(
            f"depth encoding must be 32FC1 metres, got {message.encoding!r}"
        )
    height, width, data = _shape_and_buffer(message, 4)
    dtype = np.dtype(">f4" if bool(message.is_bigendian) else "<f4")
    view = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=data,
        strides=(int(message.step), 4),
    )
    result = np.asarray(view, dtype=np.float32).copy(order="C")
    result[(~np.isfinite(result)) | (result <= 0.0)] = np.nan
    return result


def camera_matrix(camera_info: Any) -> np.ndarray:
    """Return a validated 3x3 pinhole matrix for the registered pixel grid."""
    values = np.asarray(camera_info.k, dtype=np.float64)
    if values.shape != (9,):
        raise ObservationDecodeError("CameraInfo.k must contain 9 values")
    matrix = values.reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        raise ObservationDecodeError("CameraInfo.k contains non-finite values")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ObservationDecodeError("CameraInfo fx and fy must be positive")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1.0e-9):
        raise ObservationDecodeError("CameraInfo.k has an invalid homogeneous row")
    if not np.allclose(matrix[[0, 1], [1, 0]], 0.0, atol=1.0e-12):
        raise ObservationDecodeError("CameraInfo.k must describe a zero-skew pinhole grid")
    distortion = np.asarray(camera_info.d, dtype=np.float64)
    if distortion.ndim != 1 or distortion.size == 0:
        raise ObservationDecodeError("CameraInfo.d must declare rectified distortion")
    if not np.all(np.isfinite(distortion)):
        raise ObservationDecodeError("CameraInfo.d contains non-finite values")
    if not np.allclose(distortion, 0.0, atol=1.0e-12):
        raise ObservationDecodeError(
            "CameraInfo.d must be zero for the canonical rectified RGB-D grid"
        )
    return matrix


def pose_matrix(pose_stamped: Any) -> np.ndarray:
    """Decode a ROS ``xyzw`` PoseStamped as ``T_world_camera_optical``."""
    position = pose_stamped.pose.position
    orientation = pose_stamped.pose.orientation
    translation = np.array((position.x, position.y, position.z), dtype=np.float64)
    quaternion = np.array(
        (orientation.x, orientation.y, orientation.z, orientation.w),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
        raise ObservationDecodeError("camera pose contains non-finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ObservationDecodeError("camera pose quaternion has zero norm")
    x, y, z, w = quaternion / norm
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    result[:3, 3] = translation
    return result

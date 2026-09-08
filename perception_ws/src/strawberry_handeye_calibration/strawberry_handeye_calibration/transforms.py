"""Small SE(3) helpers with explicit parent/child transform conventions."""

from __future__ import annotations

import math

import numpy as np


def validate_transform(value: np.ndarray, name: str = "transform") -> np.ndarray:
    """Return a validated copy of a rigid ``T_parent_child`` matrix."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains a non-finite value")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1.0e-6):
        raise ValueError(f"{name} rotation determinant is {determinant:.9g}, expected +1")
    return matrix.copy()


def invert_transform(value: np.ndarray) -> np.ndarray:
    """Invert one already rigid transform without a general matrix inverse."""
    matrix = validate_transform(value)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """Return the shortest angle represented by a 3x3 rotation matrix."""
    value = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return an axis-angle vector in radians, stable near zero and pi."""
    value = np.asarray(rotation, dtype=np.float64)
    angle = math.radians(rotation_angle_deg(value))
    if angle < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    if abs(math.pi - angle) < 1.0e-5:
        diagonal = np.maximum((np.diag(value) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] < 1.0e-8:
            return np.zeros(3, dtype=np.float64)
        for index in range(3):
            if index != largest:
                axis[index] = (
                    value[index, largest] + value[largest, index]
                ) / (4.0 * axis[largest])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array(
        (
            value[2, 1] - value[1, 2],
            value[0, 2] - value[2, 0],
            value[1, 0] - value[0, 1],
        ),
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    return axis * angle


def project_rotation(value: np.ndarray) -> np.ndarray:
    """Project a finite 3x3 matrix onto SO(3)."""
    matrix = np.asarray(value, dtype=np.float64)
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def mean_transform(values: list[np.ndarray]) -> np.ndarray:
    """Compute a robust center using median translation and chordal rotation."""
    if not values:
        raise ValueError("at least one transform is required")
    matrices = [validate_transform(value) for value in values]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = project_rotation(sum(item[:3, :3] for item in matrices))
    result[:3, 3] = np.median(
        np.stack([item[:3, 3] for item in matrices], axis=0), axis=0
    )
    return result


def transform_error(
    actual: np.ndarray,
    expected: np.ndarray,
) -> tuple[float, float]:
    """Return translation millimetres and rotation degrees between transforms."""
    first = validate_transform(actual)
    second = validate_transform(expected)
    translation_mm = float(np.linalg.norm(first[:3, 3] - second[:3, 3]) * 1000.0)
    rotation_deg = rotation_angle_deg(first[:3, :3] @ second[:3, :3].T)
    return translation_mm, rotation_deg


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assemble and validate a transform from OpenCV-style components."""
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return validate_transform(result)

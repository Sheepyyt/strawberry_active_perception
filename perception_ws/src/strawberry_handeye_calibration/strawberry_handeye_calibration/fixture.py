"""Deterministic offline eye-in-hand fixture and dataset generator."""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np

from .schema import CalibrationDataset, CalibrationSample, save_dataset
from .transforms import invert_transform, make_transform, project_rotation


def _axis_angle(axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    rotation, _ = cv2.Rodrigues(vector * math.radians(angle_deg))
    return rotation


def _look_at_optical(
    position: np.ndarray,
    target: np.ndarray,
    roll_deg: float,
) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    desired_down = np.array((0.0, 0.0, -1.0), dtype=np.float64)
    right = np.cross(desired_down, forward)
    if np.linalg.norm(right) < 1.0e-8:
        right = np.cross(np.array((0.0, -1.0, 0.0)), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack((right, down, forward))
    roll = _axis_angle((0.0, 0.0, 1.0), roll_deg)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = project_rotation(rotation @ roll)
    result[:3, 3] = position
    return result


def _perturb_transform(
    value: np.ndarray,
    generator: np.random.Generator,
    translation_sigma_mm: float,
    rotation_sigma_deg: float,
) -> np.ndarray:
    result = value.copy()
    if rotation_sigma_deg > 0.0:
        vector = generator.normal(
            0.0, math.radians(rotation_sigma_deg), size=3
        )
        noise_rotation, _ = cv2.Rodrigues(vector)
        result[:3, :3] = project_rotation(noise_rotation @ result[:3, :3])
    if translation_sigma_mm > 0.0:
        result[:3, 3] += generator.normal(
            0.0, translation_sigma_mm / 1000.0, size=3
        )
    return result


def make_synthetic_dataset(
    sample_count: int = 16,
    random_seed: int = 7,
    translation_noise_mm: float = 0.15,
    rotation_noise_deg: float = 0.03,
    outlier_index: int | None = None,
) -> CalibrationDataset:
    """Create a stationary-board eye-in-hand session with known ground truth."""
    if sample_count < 6:
        raise ValueError("fixture requires at least six samples")
    if outlier_index is not None and not 0 <= outlier_index < sample_count:
        raise ValueError("outlier_index is outside the generated sample range")
    if translation_noise_mm < 0.0 or rotation_noise_deg < 0.0:
        raise ValueError("fixture noise cannot be negative")

    ground_truth_link_camera = make_transform(
        _axis_angle((0.7, -0.4, 0.55), 11.0),
        np.array((0.034, -0.019, 0.071), dtype=np.float64),
    )
    ground_truth_base_board = make_transform(
        _axis_angle((0.2, 0.9, -0.3), 17.0),
        np.array((0.56, 0.02, 0.27), dtype=np.float64),
    )
    target = ground_truth_base_board[:3, 3]
    generator = np.random.default_rng(random_seed)
    samples = []
    for index in range(sample_count):
        phase = 2.0 * math.pi * index / sample_count
        ring = index % 3
        radius_xy = (0.13, 0.17, 0.20)[ring]
        position = target + np.array(
            (
                radius_xy * math.cos(phase),
                radius_xy * math.sin(phase),
                0.26 + 0.035 * math.sin(phase * 1.7),
            ),
            dtype=np.float64,
        )
        base_camera = _look_at_optical(
            position,
            target,
            roll_deg=-12.0 + 24.0 * ((index % 5) / 4.0),
        )
        base_link7 = base_camera @ invert_transform(ground_truth_link_camera)
        camera_board = invert_transform(base_camera) @ ground_truth_base_board
        camera_board = _perturb_transform(
            camera_board,
            generator,
            translation_noise_mm,
            rotation_noise_deg,
        )
        if index == outlier_index:
            outlier = make_transform(
                _axis_angle((0.3, 0.8, -0.4), 6.0),
                np.array((0.045, -0.030, 0.025), dtype=np.float64),
            )
            camera_board = outlier @ camera_board
        samples.append(
            CalibrationSample(
                sample_id=f"fixture_{index:03d}",
                timestamp_sec=1000.0 + index * 2.0,
                T_base_link7=base_link7,
                T_camera_checkerboard=camera_board,
                corner_count=88,
                reprojection_rms_px=(0.18 if index != outlier_index else 3.5),
            )
        )
    return CalibrationDataset(
        session_id="synthetic_eye_in_hand_v1",
        samples=tuple(samples),
        checkerboard_columns=11,
        checkerboard_rows=8,
        square_size_m=0.03,
        metadata={
            "source": "deterministic synthetic eye-in-hand fixture",
            "random_seed": random_seed,
            "ground_truth_T_link7_camera_optical": (
                ground_truth_link_camera.tolist()
            ),
            "ground_truth_T_base_checkerboard": ground_truth_base_board.tolist(),
            "injected_outlier_sample_id": (
                None if outlier_index is None else f"fixture_{outlier_index:03d}"
            ),
        },
    )


def main() -> None:
    """Write the deterministic offline fixture from a console command."""
    parser = argparse.ArgumentParser(
        description="Create a versioned synthetic eye-in-hand calibration dataset."
    )
    parser.add_argument("--output", required=True, help="output JSON path")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--translation-noise-mm", type=float, default=0.15)
    parser.add_argument("--rotation-noise-deg", type=float, default=0.03)
    parser.add_argument("--outlier-index", type=int)
    arguments = parser.parse_args()
    dataset = make_synthetic_dataset(
        sample_count=arguments.samples,
        random_seed=arguments.seed,
        translation_noise_mm=arguments.translation_noise_mm,
        rotation_noise_deg=arguments.rotation_noise_deg,
        outlier_index=arguments.outlier_index,
    )
    save_dataset(dataset, arguments.output)
    print(
        f"wrote {len(dataset.samples)} samples to {arguments.output}; "
        "no robot or camera process was contacted"
    )


if __name__ == "__main__":
    main()

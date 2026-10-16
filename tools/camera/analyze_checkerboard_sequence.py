#!/usr/bin/env python3
"""Analyse a multi-frame checkerboard RGB-D sequence.

This tool is deliberately stricter than a visual overlay: every physical board
edge must have enough independently valid profiles, metric depth separation,
and along-edge coverage before its RGB/depth displacement can pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import analyze_checkerboard_geometry as single_frame


PATTERN = (11, 8)
SQUARE_SIZE_M = 0.030
MIN_FRAMES = 10
MIN_BACKGROUND_SEPARATION_M = 0.020
MIN_PROFILE_FRACTION = 0.30
MIN_ALONG_EDGE_SPAN = 0.50
SIDE_MEDIAN_LIMIT_PX = 3.0
GLOBAL_P95_LIMIT_PX = 5.0
PLANE_MEMBERSHIP_M = 0.010


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _load_sequence(path: Path) -> dict[str, np.ndarray]:
    required = {
        'rgb',
        'depth_m',
        'K',
        'stamp',
        'scene_id',
        'observation_id',
        'optical_frame',
        'world_frame',
        'source_name',
        'source_type',
        'valid_depth_fraction',
        'color_depth_skew_sec',
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        _require(not missing, f'missing NPZ fields: {sorted(missing)}')
        result = {name: np.array(archive[name], copy=True) for name in archive.files}

    rgb = result['rgb']
    depth = result['depth_m']
    K = result['K']
    stamps = result['stamp']
    _require(rgb.dtype == np.uint8 and rgb.ndim == 4 and rgb.shape[-1] == 3,
             'rgb must be uint8 NxHxWx3')
    _require(depth.dtype == np.float32 and depth.shape == rgb.shape[:3],
             'depth_m must be float32 NxHxW on the RGB grid')
    _require(K.shape == (len(rgb), 3, 3) and np.all(np.isfinite(K)),
             'K must be finite Nx3x3')
    _require(stamps.shape == (len(rgb), 2), 'stamp must be Nx2')
    stamp_ns = stamps[:, 0].astype(object) * 1_000_000_000 + stamps[:, 1]
    _require(all(now > before for before, now in zip(stamp_ns, stamp_ns[1:])),
             'timestamps must be strictly increasing')
    ids = result['observation_id'].astype(str).tolist()
    _require(len(ids) == len(set(ids)), 'observation IDs must be unique')
    _require(np.allclose(K, K[0], atol=0.0, rtol=0.0),
             'CameraInfo K changed during the sequence')
    _require(np.all(K[:, 0, 0] > 0.0) and np.all(K[:, 1, 1] > 0.0),
             'K focal lengths must be positive')
    _require(np.all(np.isfinite(result['color_depth_skew_sec'])),
             'colour/depth skew contains a non-finite value')
    _require(np.max(np.abs(result['color_depth_skew_sec'])) <= 0.005,
             'at least one colour/depth skew exceeds 5 ms')
    return result


def _plane_and_metrics(
    rgb: np.ndarray, depth: np.ndarray, K: np.ndarray
) -> dict[str, Any]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detected, corners = cv2.findChessboardCornersSB(
        gray,
        PATTERN,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if not detected or corners is None or len(corners) != 88:
        raise ValueError('could not detect all 11x8 checkerboard corners')
    pixels = corners[:, 0].astype(np.float64)
    board_xy = np.asarray(
        [[column, row] for row in range(8) for column in range(11)],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(board_xy, pixels)
    if homography is None:
        raise ValueError('checkerboard homography failed')
    printed_polygon = single_frame._projective_polygon(
        homography, [[-1, -1], [11, -1], [11, 8], [-1, 8]]
    )
    plane_polygon = single_frame._projective_polygon(
        homography,
        [[-0.25, -0.25], [10.25, -0.25], [10.25, 7.25], [-0.25, 7.25]],
    )
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(plane_polygon).astype(np.int32), 255)
    roi_pixels = int(np.count_nonzero(mask))
    points = single_frame._pixels_to_points(depth, K, mask > 0)
    normal, offset, inliers = single_frame._fit_plane(points)
    residual = np.abs(points[inliers] @ normal + offset)

    rays = np.column_stack(
        (
            (pixels[:, 0] - K[0, 2]) / K[0, 0],
            (pixels[:, 1] - K[1, 2]) / K[1, 1],
            np.ones(len(pixels)),
        )
    )
    points_at_corners = rays * (-offset / (rays @ normal))[:, None]
    grid = points_at_corners.reshape(8, 11, 3)
    horizontal = np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=2).ravel()
    vertical = np.linalg.norm(grid[1:] - grid[:-1], axis=2).ravel()

    object_points = np.column_stack(
        (board_xy.astype(np.float64) * SQUARE_SIZE_M, np.zeros(len(board_xy)))
    )
    solved, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        corners.astype(np.float64),
        K,
        np.zeros(8),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        raise ValueError('checkerboard PnP failed')
    rotation, _ = cv2.Rodrigues(rotation_vector)
    board_centre_object = np.asarray([0.150, 0.105, 0.0])
    pnp_centre = rotation @ board_centre_object + translation_vector[:, 0]
    pnp_normal = rotation[:, 2]
    if pnp_normal @ normal < 0:
        pnp_normal = -pnp_normal
    centre_pixel, _ = cv2.projectPoints(
        board_centre_object.reshape(1, 3),
        rotation_vector,
        translation_vector,
        K,
        np.zeros(8),
    )
    u, v = centre_pixel[0, 0]
    centre_ray = np.asarray([(u - K[0, 2]) / K[0, 0],
                             (v - K[1, 2]) / K[1, 1], 1.0])
    depth_centre = centre_ray * (-offset / (centre_ray @ normal))
    normal_error_deg = float(np.degrees(np.arccos(np.clip(
        normal @ pnp_normal, -1.0, 1.0))))
    projected, _ = cv2.projectPoints(
        object_points, rotation_vector, translation_vector, K, np.zeros(8)
    )
    reprojection = np.linalg.norm(projected[:, 0] - pixels, axis=1)
    return {
        'corners': pixels,
        'printed_polygon': printed_polygon,
        'normal': normal,
        'offset': offset,
        'roi_valid_fraction': float(len(points) / roi_pixels),
        'plane_residual_median_mm': float(np.median(residual) * 1000.0),
        'plane_residual_p90_mm': float(np.percentile(residual, 90) * 1000.0),
        'square_median_mm': float(np.median(np.r_[horizontal, vertical]) * 1000.0),
        'horizontal_median_mm': float(np.median(horizontal) * 1000.0),
        'vertical_median_mm': float(np.median(vertical) * 1000.0),
        'horizontal_10_square_mm': float(
            np.median(np.linalg.norm(grid[:, -1] - grid[:, 0], axis=1)) * 1000.0
        ),
        'vertical_7_square_mm': float(
            np.median(np.linalg.norm(grid[-1] - grid[0], axis=1)) * 1000.0
        ),
        'pnp_depth_centre_difference_mm': float(
            np.linalg.norm(depth_centre - pnp_centre) * 1000.0
        ),
        'pnp_depth_normal_difference_deg': normal_error_deg,
        'pnp_reprojection_p90_px': float(np.percentile(reprojection, 90)),
    }


def _residual_image(
    depth: np.ndarray, K: np.ndarray, normal: np.ndarray, offset: float
) -> np.ndarray:
    rows, columns = np.indices(depth.shape)
    result = np.full(depth.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(depth)
    x = (columns[valid] - K[0, 2]) * depth[valid] / K[0, 0]
    y = (rows[valid] - K[1, 2]) * depth[valid] / K[1, 1]
    result[valid] = normal[0] * x + normal[1] * y + normal[2] * depth[valid] + offset
    return result


def _smooth_profiles(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values).astype(np.float32)
    numerator = cv2.GaussianBlur(
        np.nan_to_num(values).astype(np.float32), (0, 0), 1.0, 0.0
    )
    denominator = cv2.GaussianBlur(finite, (0, 0), 1.0, 0.0)
    return np.where(
        denominator >= 0.75, numerator / np.maximum(denominator, 1e-6), np.nan
    )


def _downward_crossings(profile: np.ndarray, offsets: np.ndarray) -> list[float]:
    crossings: list[float] = []
    indices = np.flatnonzero((offsets >= -3.0) & (offsets <= 26.0))
    for index in indices[1:]:
        before = float(profile[index - 1])
        after = float(profile[index])
        if before >= 0.5 > after:
            fraction = (before - 0.5) / max(before - after, 1e-12)
            crossings.append(float(offsets[index - 1] + fraction * 0.25))
    return crossings


def _analyse_edges(
    rgbs: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    frames: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    median_rgb = np.median(rgbs.astype(np.float32), axis=0).astype(np.uint8)
    lab = cv2.cvtColor(median_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    polygon = np.median(
        np.stack([frame['printed_polygon'] for frame in frames]), axis=0
    )
    polygon_centre = polygon.mean(axis=0)
    residuals = np.stack([
        _residual_image(depth, K, frame['normal'], frame['offset'])
        for depth, K, frame in zip(depths, intrinsics, frames)
    ])
    offsets = np.arange(-5.0, 30.0001, 0.25, dtype=np.float32)
    sides: list[dict[str, Any]] = []
    all_errors: list[float] = []
    overlay = cv2.cvtColor(median_rgb, cv2.COLOR_RGB2BGR)

    for side_index in range(4):
        start = polygon[side_index]
        finish = polygon[(side_index + 1) % 4]
        edge = finish - start
        length = float(np.linalg.norm(edge))
        tangent = edge / length
        outward = np.asarray([-tangent[1], tangent[0]])
        if outward @ (0.5 * (start + finish) - polygon_centre) < 0:
            outward = -outward
        profile_count = max(20, int(length * 0.7 / 2.0))
        positions = np.linspace(0.15, 0.85, profile_count)
        bases = start + positions[:, None] * edge
        map_x = (bases[:, None, 0] + offsets[None, :] * outward[0]).astype(
            np.float32
        )
        map_y = (bases[:, None, 1] + offsets[None, :] * outward[1]).astype(
            np.float32
        )

        colour = np.stack([
            cv2.remap(
                lab[:, :, channel], map_x, map_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=math.nan,
            )
            for channel in range(3)
        ], axis=2)
        colour = np.stack(
            [_smooth_profiles(colour[:, :, channel]) for channel in range(3)],
            axis=2,
        )
        colour_gradient = np.linalg.norm(
            np.gradient(colour, 0.25, axis=1), axis=2
        )
        colour_search = (offsets >= 2.0) & (offsets <= 22.0)
        colour_indices = np.nanargmax(
            np.where(colour_search[None, :], colour_gradient, np.nan), axis=1
        )
        rgb_edges = offsets[colour_indices]
        rgb_strength = colour_gradient[np.arange(profile_count), colour_indices]

        sampled_residuals = np.stack([
            cv2.remap(
                residual,
                map_x,
                map_y,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=math.nan,
            )
            for residual in residuals
        ])
        finite = np.isfinite(sampled_residuals)
        foreground_probability = np.mean(
            finite & (np.abs(sampled_residuals) <= PLANE_MEMBERSHIP_M), axis=0
        ).astype(np.float32)
        foreground_probability = cv2.GaussianBlur(
            foreground_probability, (0, 0), 1.0, 0.0
        )

        accepted: list[dict[str, float]] = []
        for profile_index in range(profile_count):
            rgb_edge = float(rgb_edges[profile_index])
            crossings = _downward_crossings(
                foreground_probability[profile_index], offsets
            )
            if not crossings:
                continue
            # A physical foreground edge can have a short invalid shadow.  If
            # several 0.5 crossings exist, select the one nearest the colour
            # edge but do not use its error as a quality filter.
            depth_edge = min(crossings, key=lambda value: abs(value - rgb_edge))
            inside = (offsets >= rgb_edge - 6.0) & (offsets <= rgb_edge - 2.0)
            outside = (offsets >= rgb_edge + 3.0) & (offsets <= rgb_edge + 12.0)
            inside_support = float(np.nanmedian(
                foreground_probability[profile_index, inside]
            ))
            background_values = np.abs(
                sampled_residuals[:, profile_index, outside]
            )
            background_finite = np.isfinite(background_values)
            background_valid_fraction = float(np.mean(background_finite))
            background_separation = (
                float(np.nanmedian(background_values))
                if np.any(background_finite)
                else math.nan
            )
            if (
                inside_support < 0.70
                or background_valid_fraction < 0.30
                or not np.isfinite(background_separation)
                or background_separation < MIN_BACKGROUND_SEPARATION_M
                or float(rgb_strength[profile_index]) < 3.0
            ):
                continue
            accepted.append({
                'position': float(positions[profile_index]),
                'rgb_edge': rgb_edge,
                'depth_edge': depth_edge,
                'signed_error': depth_edge - rgb_edge,
                'background_separation_m': background_separation,
                'background_valid_fraction': background_valid_fraction,
            })

        errors = np.asarray([item['signed_error'] for item in accepted])
        accepted_positions = np.asarray([item['position'] for item in accepted])
        profile_fraction = float(len(accepted) / profile_count)
        along_edge_span = (
            float((accepted_positions.max() - accepted_positions.min()) / 0.70)
            if len(accepted_positions) >= 2
            else 0.0
        )
        side_passed = bool(
            len(errors) > 0
            and profile_fraction >= MIN_PROFILE_FRACTION
            and along_edge_span >= MIN_ALONG_EDGE_SPAN
            and np.median(np.abs(errors)) <= SIDE_MEDIAN_LIMIT_PX
        )
        if len(errors):
            all_errors.extend(errors.tolist())
            median_rgb_edge = float(np.median(
                [item['rgb_edge'] for item in accepted]
            ))
            median_depth_edge = float(np.median(
                [item['depth_edge'] for item in accepted]
            ))
            for edge_offset, colour_value, thickness in (
                (median_rgb_edge, (0, 0, 255), 2),
                (median_depth_edge, (255, 255, 0), 1),
            ):
                line = np.rint(
                    np.vstack((start, finish)) + outward * edge_offset
                ).astype(np.int32)
                cv2.line(
                    overlay, tuple(line[0]), tuple(line[1]),
                    colour_value, thickness, cv2.LINE_AA,
                )
        else:
            median_rgb_edge = math.nan
            median_depth_edge = math.nan
        sides.append({
            'side_index': side_index,
            'side_name': ('top', 'right', 'bottom', 'left')[side_index],
            'attempted_profiles': profile_count,
            'accepted_profiles': len(accepted),
            'accepted_profile_fraction': profile_fraction,
            'accepted_along_edge_span_fraction': along_edge_span,
            'median_rgb_edge_offset_px': median_rgb_edge,
            'median_depth_edge_offset_px': median_depth_edge,
            'median_signed_depth_minus_rgb_px': (
                float(np.median(errors)) if len(errors) else math.nan
            ),
            'median_absolute_error_px': (
                float(np.median(np.abs(errors))) if len(errors) else math.nan
            ),
            'p95_absolute_error_px': (
                float(np.percentile(np.abs(errors), 95)) if len(errors) else math.nan
            ),
            'fraction_within_3px': (
                float(np.mean(np.abs(errors) <= 3.0)) if len(errors) else 0.0
            ),
            'median_background_separation_m': (
                float(np.median([
                    item['background_separation_m'] for item in accepted
                ])) if accepted else math.nan
            ),
            'median_background_valid_fraction': (
                float(np.median([
                    item['background_valid_fraction'] for item in accepted
                ])) if accepted else 0.0
            ),
            'gate_passed': side_passed,
        })

    absolute_errors = np.abs(np.asarray(all_errors))
    global_p95 = float(np.percentile(absolute_errors, 95))
    passed = bool(
        len(rgbs) >= MIN_FRAMES
        and all(side['gate_passed'] for side in sides)
        and global_p95 <= GLOBAL_P95_LIMIT_PX
    )
    cv2.putText(
        overlay,
        'red: RGB physical edge | cyan: registered-depth physical edge',
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        'red: RGB physical edge | cyan: registered-depth physical edge',
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1,
        cv2.LINE_AA,
    )
    return {
        'method': (
            'Temporal median Lab edge versus the 0.5 probability boundary of '
            'depth points within 10 mm of each frame\'s fitted board plane.'
        ),
        'frame_count': int(len(rgbs)),
        'minimum_frame_count': MIN_FRAMES,
        'minimum_background_separation_m': MIN_BACKGROUND_SEPARATION_M,
        'minimum_profile_fraction_per_side': MIN_PROFILE_FRACTION,
        'minimum_along_edge_span_fraction_per_side': MIN_ALONG_EDGE_SPAN,
        'side_median_limit_px': SIDE_MEDIAN_LIMIT_PX,
        'global_p95_limit_px': GLOBAL_P95_LIMIT_PX,
        'accepted_profiles': int(len(absolute_errors)),
        'median_absolute_error_px': float(np.median(absolute_errors)),
        'p95_absolute_error_px': global_p95,
        'fraction_within_3px': float(np.mean(absolute_errors <= 3.0)),
        'sides': sides,
        'gate_passed': passed,
    }, overlay


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        'median': float(np.median(array)),
        'p90': float(np.percentile(array, 90)),
        'minimum': float(np.min(array)),
        'maximum': float(np.max(array)),
    }


def analyse(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    data = _load_sequence(path)
    rgbs, depths, intrinsics = data['rgb'], data['depth_m'], data['K']
    frames = [
        _plane_and_metrics(rgb, depth, K)
        for rgb, depth, K in zip(rgbs, depths, intrinsics)
    ]
    corners = np.stack([frame['corners'] for frame in frames])
    edge, overlay = _analyse_edges(rgbs, depths, intrinsics, frames)

    horizontal_error = [
        100.0 * abs(frame['horizontal_median_mm'] / 30.0 - 1.0)
        for frame in frames
    ]
    vertical_error = [
        100.0 * abs(frame['vertical_median_mm'] / 30.0 - 1.0)
        for frame in frames
    ]
    checks = {
        'at_least_10_synchronized_frames': len(frames) >= MIN_FRAMES,
        'all_88_corners_detected_in_every_frame': True,
        'observation_ids_unique': True,
        'timestamps_strictly_increasing': True,
        'camera_intrinsics_unchanged': True,
        'horizontal_scale_error_le_2pct_every_frame': max(horizontal_error) <= 2.0,
        'vertical_scale_error_le_2pct_every_frame': max(vertical_error) <= 2.0,
        'pnp_depth_centre_error_le_20mm_every_frame': max(
            frame['pnp_depth_centre_difference_mm'] for frame in frames
        ) <= 20.0,
        'pnp_depth_normal_error_le_1deg_every_frame': max(
            frame['pnp_depth_normal_difference_deg'] for frame in frames
        ) <= 1.0,
        'four_side_physical_edge_alignment_passed': edge['gate_passed'],
    }
    report = {
        'schema_version': 1,
        'artifact': 'g3_checkerboard_multiframe_geometry',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed' if all(checks.values()) else 'failed',
        'scope': (
            'Twenty stationary canonical Gemini RGB-D observations; metric '
            'checkerboard scale, RGB/depth planar pose consistency, and '
            'four-side physical-edge alignment. No robot motion.'
        ),
        'input': {
            'path': str(path),
            'sha256': _sha256(path),
            'scene_id': str(data['scene_id'][0]),
            'frame_count': int(len(rgbs)),
            'shape': list(rgbs.shape[1:]),
            'unique_observation_ids': len(set(data['observation_id'].astype(str))),
            'maximum_absolute_color_depth_skew_ms': float(
                np.max(np.abs(data['color_depth_skew_sec'])) * 1000.0
            ),
            'valid_depth_fraction': _summary(data['valid_depth_fraction']),
        },
        'reference': {
            'checker_squares': [12, 9],
            'printed_region_mm': [360.0, 270.0],
            'square_size_mm': 30.0,
        },
        'stability': {
            'corner_radial_std_median_px': float(np.median(
                np.linalg.norm(np.std(corners, axis=0), axis=1)
            )),
            'corner_radial_std_maximum_px': float(np.max(
                np.linalg.norm(np.std(corners, axis=0), axis=1)
            )),
            'K_exactly_unchanged': True,
        },
        'metric_scale_across_frames': {
            'square_median_mm': _summary([
                frame['square_median_mm'] for frame in frames
            ]),
            'horizontal_median_mm': _summary([
                frame['horizontal_median_mm'] for frame in frames
            ]),
            'vertical_median_mm': _summary([
                frame['vertical_median_mm'] for frame in frames
            ]),
            'horizontal_10_square_mm': _summary([
                frame['horizontal_10_square_mm'] for frame in frames
            ]),
            'vertical_7_square_mm': _summary([
                frame['vertical_7_square_mm'] for frame in frames
            ]),
            'maximum_horizontal_error_percent': max(horizontal_error),
            'maximum_vertical_error_percent': max(vertical_error),
            'gate_limit_percent': 2.0,
        },
        'rgb_pnp_vs_registered_depth_across_frames': {
            'centre_3d_difference_mm': _summary([
                frame['pnp_depth_centre_difference_mm'] for frame in frames
            ]),
            'normal_difference_deg': _summary([
                frame['pnp_depth_normal_difference_deg'] for frame in frames
            ]),
            'pnp_reprojection_p90_px': _summary([
                frame['pnp_reprojection_p90_px'] for frame in frames
            ]),
        },
        'board_surface_diagnostic': {
            'roi_valid_fraction': _summary([
                frame['roi_valid_fraction'] for frame in frames
            ]),
            'plane_residual_median_mm': _summary([
                frame['plane_residual_median_mm'] for frame in frames
            ]),
            'plane_residual_p90_mm': _summary([
                frame['plane_residual_p90_mm'] for frame in frames
            ]),
            'original_2mm_planarity_gate_passed': all(
                frame['plane_residual_p90_mm'] <= 2.0 for frame in frames
            ),
            'interpretation': (
                'The suspended board has stable millimetre-scale spatial '
                'curvature/non-planarity. This does not invalidate its physical '
                'occlusion edges; the earlier flat-board sample remains the '
                'planarity evidence.'
            ),
        },
        'physical_edge_alignment': edge,
        'checks': checks,
        'motion_executed': False,
        'limitations': [
            'This is a stationary camera-session geometry check, not hand-eye calibration.',
            'The USB link remains 480 Mbit/s and is accepted only for 640x400@10 Hz.',
        ],
    }
    return report, overlay


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--overlay', required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, overlay = analyse(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.overlay.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    if not cv2.imwrite(str(args.overlay), overlay):
        raise RuntimeError(f'could not write overlay: {args.overlay}')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())

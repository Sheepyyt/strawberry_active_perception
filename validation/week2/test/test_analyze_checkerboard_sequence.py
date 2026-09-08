"""Unit tests for multi-frame checkerboard helpers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


WEEK2 = Path(__file__).parents[1]
sys.path.insert(0, str(WEEK2))
SPEC = importlib.util.spec_from_file_location(
    'checkerboard_sequence_analysis', WEEK2 / 'analyze_checkerboard_sequence.py'
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_downward_crossing_is_subpixel() -> None:
    offsets = np.arange(-3.0, 5.0, 0.25, dtype=np.float32)
    probability = np.clip((2.75 - offsets) / 2.0 + 0.5, 0.0, 1.0)

    crossings = MODULE._downward_crossings(probability, offsets)

    assert crossings == pytest.approx([2.75])


def test_downward_crossing_does_not_invent_an_edge() -> None:
    offsets = np.arange(-3.0, 5.0, 0.25, dtype=np.float32)
    assert MODULE._downward_crossings(np.ones_like(offsets), offsets) == []


def test_summary_reports_distribution_extrema() -> None:
    summary = MODULE._summary([1.0, 2.0, 3.0])
    assert summary['median'] == 2.0
    assert summary['minimum'] == 1.0
    assert summary['maximum'] == 3.0


def test_load_sequence_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / 'duplicate.npz'
    np.savez_compressed(
        path,
        rgb=np.zeros((2, 2, 3, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2, 3), dtype=np.float32),
        K=np.repeat(np.eye(3)[None], 2, axis=0),
        stamp=np.asarray([[1, 0], [2, 0]], dtype=np.int64),
        scene_id=np.asarray(['scene', 'scene']),
        observation_id=np.asarray(['same', 'same']),
        optical_frame=np.asarray(['camera', 'camera']),
        world_frame=np.asarray(['world', 'world']),
        source_name=np.asarray(['fixture', 'fixture']),
        source_type=np.asarray([1, 1], dtype=np.uint8),
        valid_depth_fraction=np.asarray([1.0, 1.0], dtype=np.float32),
        color_depth_skew_sec=np.asarray([0.0, 0.0]),
    )
    with pytest.raises(ValueError, match='unique'):
        MODULE._load_sequence(path)

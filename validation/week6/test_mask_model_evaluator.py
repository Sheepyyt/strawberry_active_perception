from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("mask_model_evaluator.py")
SPEC = importlib.util.spec_from_file_location("mask_model_evaluator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_union_keeps_only_selected_labels_above_confidence() -> None:
    masks = np.zeros((3, 4, 5), dtype=np.float32)
    masks[0, 1:3, 1:3] = 0.9
    masks[1, :, :] = 1.0
    masks[2, 0, 0] = 1.0
    output, accepted = MODULE.union_selected_instance_masks(
        masks=masks,
        class_ids=[9, 8, 5],
        confidences=[0.8, 0.9, 0.2],
        names={9: "Healthy Strawberry", 8: "Healthy Leaf", 5: "gray_mold"},
        allowed_labels=["Healthy Strawberry", "gray_mold"],
        confidence_threshold=0.35,
        output_shape=(4, 5),
    )
    assert output.dtype == np.uint8
    assert set(np.unique(output)) <= {0, 255}
    assert np.count_nonzero(output) == 4
    assert [item["label"] for item in accepted] == ["Healthy Strawberry"]


def test_union_resizes_masks_with_nearest_neighbour() -> None:
    masks = np.zeros((1, 2, 2), dtype=np.float32)
    masks[0, 0, 0] = 1.0
    output, accepted = MODULE.union_selected_instance_masks(
        masks=masks,
        class_ids=[0],
        confidences=[1.0],
        names=["fruit"],
        allowed_labels=["fruit"],
        confidence_threshold=0.5,
        output_shape=(4, 4),
    )
    assert np.count_nonzero(output) == 4
    assert len(accepted) == 1


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_union_rejects_bad_confidence_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="confidence threshold"):
        MODULE.union_selected_instance_masks(
            masks=np.empty((0, 2, 2), dtype=np.float32),
            class_ids=[],
            confidences=[],
            names=[],
            allowed_labels=[],
            confidence_threshold=threshold,
            output_shape=(2, 2),
        )


def test_union_rejects_mismatched_instance_arrays() -> None:
    with pytest.raises(ValueError, match="counts must match"):
        MODULE.union_selected_instance_masks(
            masks=np.zeros((1, 2, 2), dtype=np.float32),
            class_ids=[],
            confidences=[],
            names=[],
            allowed_labels=[],
            confidence_threshold=0.5,
            output_shape=(2, 2),
        )

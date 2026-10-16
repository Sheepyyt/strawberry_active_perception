"""Frozen first-step endpoints tolerate measured jitter without changing targets."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from strawberry_active_perception_bridge.real_nbv_contract import (
    LARGE_MOTION_LIMITS,
    MotionBudgetExhausted,
    MotionSessionLedger,
    camera_motion,
    validate_frozen_next_view,
    validate_raw_next_view,
)


# Saved evidence from run_20261016_223939_75225c. No live ROS or run files are
# needed: encoder jitter shifted the measured start 19.45 um, making the exact
# frozen 100 mm endpoint appear 8.714 um beyond the radius at the fresh start.
FROZEN_START = np.array([
    [-0.6131668134119184, -0.02837283071353062, -0.7894437544293927, 0.13686054113830556],
    [0.7899523067373471, -0.023726073659948967, -0.6127090879928476, 0.04672422309799066],
    [-0.0013461094377155553, -0.9993157938839876, 0.03696122405166913, 0.3485079436120534],
    [0.0, 0.0, 0.0, 1.0],
])
MEASURED_START = np.array([
    [-0.6131519416735635, -0.028394785181161925, -0.7894545158503146, 0.13687043693948164],
    [0.789963990567477, -0.023634520637572605, -0.6126975624571552, 0.04670835093464156],
    [-0.0012609633802966387, -0.9993173397918942, 0.036922410032468475, 0.34850260742346706],
    [0.0, 0.0, 0.0, 1.0],
])
FROZEN_TARGET = np.array([
    [-0.4747059806295687, 0.13766097994551388, -0.8693121916520903, 0.05158185176507794],
    [0.8801131779076897, 0.06592044357358415, -0.47016517224512644, 0.031639623707070214],
    [-0.007417953069885033, -0.9882833347373126, -0.15245007134386734, 0.39850794361205344],
    [0.0, 0.0, 0.0, 1.0],
])


def ledger(**kwargs):
    return MotionSessionLedger(
        max_motion_steps=8, initial_camera=FROZEN_START, initial_coverage=.088944,
        motion_limits=LARGE_MOTION_LIMITS, minimum_target_pixels=100, **kwargs,
    )


def test_recorded_jitter_passes_with_exact_frozen_endpoint_and_hash_unchanged():
    target = FROZEN_TARGET.copy()
    before = hashlib.sha256(target.tobytes()).hexdigest()
    with pytest.raises(ValueError, match="numerical tolerance"):
        validate_raw_next_view(MEASURED_START, target, gain=.049, max_step_m=.1)
    motion = validate_frozen_next_view(
        FROZEN_START, MEASURED_START, target, gain=.049, max_step_m=.1
    )
    assert camera_motion(FROZEN_START, target).translation_m == pytest.approx(.1)
    drift = camera_motion(FROZEN_START, MEASURED_START).translation_m
    assert drift == pytest.approx(19.45063903644034e-6)
    assert motion.translation_m - .1 == pytest.approx(8.7143635194e-6)
    assert hashlib.sha256(target.tobytes()).hexdigest() == before
    np.testing.assert_array_equal(target, FROZEN_TARGET)


def test_jitter_allowance_cannot_rescue_an_originally_overlong_plan():
    target = FROZEN_TARGET.copy()
    displacement = target[:3, 3] - FROZEN_START[:3, 3]
    target[:3, 3] = FROZEN_START[:3, 3] + displacement * 1.0001
    with pytest.raises(ValueError, match="numerical tolerance"):
        validate_frozen_next_view(
            FROZEN_START, MEASURED_START, target, gain=.049, max_step_m=.1
        )


def test_one_millimetre_start_gate_still_rejects_real_movement():
    moved = FROZEN_START.copy()
    moved[0, 3] += .001001
    with pytest.raises(ValueError, match="start translation gate"):
        validate_frozen_next_view(
            FROZEN_START, moved, FROZEN_TARGET, gain=.049, max_step_m=.1
        )
    with pytest.raises(ValueError, match="allowance must be"):
        validate_frozen_next_view(
            FROZEN_START, MEASURED_START, FROZEN_TARGET, gain=.049,
            max_step_m=.1, max_start_translation_drift_m=.002,
        )


def test_frozen_start_rotation_gate_is_independently_checked():
    moved = FROZEN_START.copy()
    angle = np.radians(.6)
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    moved[:3, :3] = rotation @ moved[:3, :3]
    with pytest.raises(ValueError, match="start rotation gate"):
        validate_frozen_next_view(
            FROZEN_START, moved, FROZEN_TARGET, gain=.049, max_step_m=.1
        )


def test_first_step_passes_both_ledger_checks_and_records_measured_length():
    book = ledger()
    args = dict(planned_start_camera=MEASURED_START, planned_camera=FROZEN_TARGET)
    with pytest.raises(ValueError, match="outside the selected motion profile"):
        book.validate_next_planned_step(**args)
    planned = book.validate_next_planned_step(**args, frozen_start_camera=FROZEN_START)
    result = book.record_closed_step(
        **args, frozen_start_camera=FROZEN_START, actual_camera=FROZEN_TARGET,
        coverage_after=.15, target_valid_mask_depth_pixels=400,
        reported_motion_goal_count=1, gates_closed=True,
    )
    assert result.planned_step_translation_m == planned.translation_m
    assert result.planned_step_translation_m == pytest.approx(.1000087143635194)
    assert book.motion_goal_count == 1
    # The first-step evidence cannot loosen a later dynamic target.
    with pytest.raises(ValueError, match="only valid for the first step"):
        book.validate_next_planned_step(**args, frozen_start_camera=FROZEN_START)


def test_frozen_start_reference_must_be_the_bound_session_start():
    wrong_reference = FROZEN_START.copy()
    wrong_reference[0, 3] += .0001
    with pytest.raises(ValueError, match="session's initial camera"):
        ledger().validate_next_planned_step(
            planned_start_camera=MEASURED_START, planned_camera=FROZEN_TARGET,
            frozen_start_camera=wrong_reference,
        )


def test_cumulative_budget_is_not_extended_by_start_drift():
    book = MotionSessionLedger(
        max_motion_steps=1, initial_camera=FROZEN_START, initial_coverage=.08,
        motion_limits=replace(LARGE_MOTION_LIMITS, maximum_session_translation_m=.1),
    )
    with pytest.raises(MotionBudgetExhausted, match="cumulative"):
        book.validate_next_planned_step(
            planned_start_camera=MEASURED_START, planned_camera=FROZEN_TARGET,
            frozen_start_camera=FROZEN_START,
        )

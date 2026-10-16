"""Acceptance checks for bounded persistent-map NBV sessions."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from strawberry_active_perception_bridge.real_nbv_contract import (
    MotionSessionLedger,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT / "perception_ws" / "src"
    / "strawberry_active_perception_bridge" / "config"
    / "real_nbv_supervisor.yaml"
)
SUPERVISOR = (
    ROOT / "perception_ws" / "src"
    / "strawberry_active_perception_bridge"
    / "strawberry_active_perception_bridge" / "real_nbv_supervisor.py"
)


def _camera(x_m: float) -> np.ndarray:
    transform = np.eye(4)
    transform[0, 3] = x_m
    return transform


def test_default_remains_one_step_and_bounded_convergence_is_explicit() -> None:
    values = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[
        "real_nbv_supervisor"
    ]["ros__parameters"]
    assert values["max_motion_steps"] == 1
    assert values["coverage_target"] == 0.0
    assert values["coverage_plateau_delta"] == pytest.approx(0.005)
    assert values["coverage_plateau_patience"] == 2
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert "EXECUTE_REAL_NBV_SESSION_" in source
    assert source.count("move_client.send_goal_async(goal)") == 1
    assert "ConfigureNBV may be called only once" in source


@pytest.mark.parametrize(
    "fault,gates_closed",
    (
        ("second_step_ik_failed", True),
        ("target_lost", True),
        ("duplicate_observation", True),
        ("coverage_decreased", True),
        ("gate_close_failed", False),
        ("extra_command_publisher", True),
    ),
)
def test_injected_fault_latches_reason_and_blocks_later_goals(
    tmp_path: Path, fault: str, gates_closed: bool
) -> None:
    ledger = MotionSessionLedger(
        max_motion_steps=3,
        initial_camera=np.eye(4),
        initial_coverage=0.30,
    )
    ledger.record_closed_step(
        planned_camera=_camera(0.004),
        actual_camera=_camera(0.0039),
        coverage_after=0.36,
        target_valid_mask_depth_pixels=400,
        reported_motion_goal_count=1,
        gates_closed=True,
    )
    ledger.abort(fault, gates_closed=gates_closed)
    artifact = tmp_path / f"{fault}.json"
    artifact.write_text(
        json.dumps(ledger.summary(), sort_keys=True), encoding="utf-8"
    )
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["motion_goal_count"] == 1
    assert saved["termination_reason"] == fault
    assert saved["gates_closed_at_termination"] is gates_closed
    with pytest.raises(ValueError, match="terminated session"):
        ledger.record_closed_step(
            planned_camera=_camera(0.008),
            actual_camera=_camera(0.008),
            coverage_after=0.42,
            target_valid_mask_depth_pixels=400,
            reported_motion_goal_count=2,
            gates_closed=True,
        )

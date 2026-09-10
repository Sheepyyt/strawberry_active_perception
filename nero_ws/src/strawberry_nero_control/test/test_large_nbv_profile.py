"""Keep the 50--100 mm NBV controller overlay narrow and explicit."""

from pathlib import Path

import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def _parameters(path: Path) -> dict[str, object]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document["/**"]["ros__parameters"]


def test_large_nbv_overlay_changes_only_precision_joint_delta() -> None:
    base = _parameters(PACKAGE / "config/nero_control.yaml")
    overlay = _parameters(PACKAGE / "config/large_nbv_experiment.yaml")
    assert base["max_joint_delta_rad"] == 0.35
    assert base["precision_max_joint_delta_rad"] == 0.12
    assert overlay == {"precision_max_joint_delta_rad": 0.35}
    assert "trajectory_velocity_limits" not in overlay
    assert "ik_position_tolerance_m" not in overlay
    assert "final_position_tolerance_m" not in overlay

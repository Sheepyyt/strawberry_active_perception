"""Pure tests for the reachability-aware candidate plot."""

from pathlib import Path

import numpy as np

from reachable_candidate_visualization import render_candidate_lattice


def _pose(x: float, y: float, z: float):
    pose = np.eye(4)
    pose[:3, 3] = (x, y, z)
    return pose.tolist()


def test_candidate_plot_separates_rejected_useful_and_selected(tmp_path: Path) -> None:
    document = {
        "final_tf": {"T_base_camera_optical": _pose(0.0, 0.0, 0.5)},
        "final_target": {"base_xyz_m": [0.0, 0.0, 0.0]},
        "selected_candidate": {"T_base_camera": _pose(0.1, 0.0, 0.5)},
        "ik_candidates": [
            {
                "T_base_camera": _pose(-0.1, 0.0, 0.5),
                "ik_success": False,
                "ik_gate_passed": False,
                "rejection": "unreachable",
            },
            {
                "T_base_camera": _pose(0.0, 0.1, 0.5),
                "ik_success": True,
                "ik_gate_passed": True,
                "rejection": "candidate does not improve map information gain",
                "gain_scored": True,
                "gain_improvement": 0.0,
            },
            {
                "T_base_camera": _pose(0.1, 0.0, 0.5),
                "ik_success": True,
                "ik_gate_passed": True,
                "rejection": "",
                "gain_scored": True,
                "gain_improvement": 0.2,
            },
        ],
    }
    output = tmp_path / "candidates.png"
    counts = render_candidate_lattice(document, output)
    assert counts == {
        "IK rejected": 1,
        "reachable, no gain": 1,
        "reachable + useful": 1,
    }
    assert output.is_file() and output.stat().st_size > 1000

#!/usr/bin/env python3
"""Keep exhibition relaxations out of the research controller profile."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = (
    ROOT / "nero_ws" / "src" / "strawberry_nero_control"
    / "config" / "nero_control.yaml"
)
EXHIBITION = ROOT / "nero_exhibition_demo" / "config" / "nero_exhibition.yaml"
START_SCRIPT = ROOT / "nero_exhibition_demo" / "start_control.sh"


def parameters(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["/**"][
        "ros__parameters"
    ]


class ProfileIsolationTests(unittest.TestCase):
    def test_research_profile_remains_strict(self) -> None:
        values = parameters(RESEARCH)
        self.assertEqual(values["ik_position_tolerance_m"], 0.002)
        self.assertAlmostEqual(values["ik_orientation_tolerance_rad"], 0.034906585)
        self.assertEqual(values["max_joint_delta_rad"], 0.35)
        self.assertEqual(values["final_position_tolerance_m"], 0.010)

    def test_exhibition_relaxations_are_explicit_and_local(self) -> None:
        values = parameters(EXHIBITION)
        self.assertEqual(values["ik_position_tolerance_m"], 0.050)
        self.assertAlmostEqual(values["ik_orientation_tolerance_rad"], 0.087266463)
        self.assertEqual(values["max_joint_delta_rad"], 1.50)
        self.assertEqual(values["final_position_tolerance_m"], 0.050)

    def test_exhibition_launcher_selects_only_its_overlay(self) -> None:
        script = START_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            'profile_config_file:="${demo_directory}/config/nero_exhibition.yaml"',
            script,
        )


if __name__ == "__main__":
    unittest.main()

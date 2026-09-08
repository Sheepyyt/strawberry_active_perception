"""Command-line entry point for offline calibration and JSON reporting."""

from __future__ import annotations

import argparse
import json

from .calibration import CalibrationConfig, calibrate_eye_in_hand
from .schema import load_dataset, write_json_atomic


def main() -> None:
    """Calibrate one saved session without starting ROS or robot control."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate T_link7_camera_optical from synchronized link7 and "
            "checkerboard poses. This command never contacts the robot."
        )
    )
    parser.add_argument("--input", required=True, help="versioned sample JSON")
    parser.add_argument("--output", required=True, help="calibration report JSON")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"),
    )
    arguments = parser.parse_args()
    dataset = load_dataset(arguments.input)
    report = calibrate_eye_in_hand(
        dataset,
        CalibrationConfig(
            methods=tuple(arguments.methods),
            holdout_fraction=arguments.holdout_fraction,
            random_seed=arguments.seed,
        ),
    )
    write_json_atomic(report, arguments.output)
    print(
        json.dumps(
            {
                "success": report["success"],
                "status": report["status"],
                "reason": report["reason"],
                "selected_method": report.get("selected_method"),
                "output": arguments.output,
            },
            ensure_ascii=False,
        )
    )
    if not report["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

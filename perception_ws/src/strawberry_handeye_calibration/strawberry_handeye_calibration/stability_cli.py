"""CLI for deterministic multi-holdout hand-eye stability acceptance."""

from __future__ import annotations

import argparse
import json

from .calibration import CalibrationConfig
from .schema import load_dataset, write_json_atomic
from .stability import StabilityConfig, validate_cross_split_stability


def main() -> None:
    """Validate one saved dataset across many splits without contacting hardware."""
    parser = argparse.ArgumentParser(
        description=(
            "Repeat offline hand-eye calibration across deterministic holdout "
            "splits and reject unstable transforms. This never contacts the robot."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-count", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--max-transform-translation-mm", type=float, default=10.0)
    parser.add_argument("--max-transform-rotation-deg", type=float, default=2.0)
    arguments = parser.parse_args()
    if arguments.split_count < 2:
        parser.error("--split-count must be at least 2")
    seeds = tuple(
        range(arguments.seed_start, arguments.seed_start + arguments.split_count)
    )
    report = validate_cross_split_stability(
        load_dataset(arguments.input),
        CalibrationConfig(holdout_fraction=arguments.holdout_fraction),
        StabilityConfig(
            split_seeds=seeds,
            maximum_pairwise_translation_mm=(
                arguments.max_transform_translation_mm
            ),
            maximum_pairwise_rotation_deg=arguments.max_transform_rotation_deg,
        ),
    )
    write_json_atomic(report, arguments.output)
    print(
        json.dumps(
            {
                "success": report["success"],
                "status": report["status"],
                "reason": report["reason"],
                "successful_splits": report["successful_split_count"],
                "split_count": report["split_count"],
                "recommended_method": report["recommended_method"],
                "safe_for_robot_use": report["safe_for_robot_use"],
                "output": arguments.output,
            },
            ensure_ascii=False,
        )
    )
    if not report["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""CLI for an auditable frozen prospective hand-eye check."""

from __future__ import annotations

import argparse
import json

from .prospective import validate_prospective_file
from .schema import write_json_atomic


def main() -> None:
    """Fit declared training samples once and score declared later samples."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen train/test hand-eye diagnostic from saved JSON. "
            "This command never starts ROS or contacts robot hardware."
        )
    )
    parser.add_argument("--input", required=True, help="versioned sample JSON")
    parser.add_argument("--output", required=True, help="audit report JSON")
    parser.add_argument(
        "--train-ids",
        nargs="+",
        required=True,
        help="ordered sample IDs used for the one-time robust fit",
    )
    parser.add_argument(
        "--test-ids",
        nargs="+",
        required=True,
        help="ordered later sample IDs used only for frozen evaluation",
    )
    parser.add_argument("--max-translation-p95-mm", type=float, default=10.0)
    parser.add_argument("--max-rotation-p95-deg", type=float, default=2.0)
    arguments = parser.parse_args()
    report = validate_prospective_file(
        arguments.input,
        arguments.train_ids,
        arguments.test_ids,
        maximum_translation_p95_mm=arguments.max_translation_p95_mm,
        maximum_rotation_p95_deg=arguments.max_rotation_p95_deg,
    )
    write_json_atomic(report, arguments.output)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "status": report["status"],
                "reason": report["reason"],
                "safe_for_robot_use": report["safe_for_robot_use"],
                "output": arguments.output,
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

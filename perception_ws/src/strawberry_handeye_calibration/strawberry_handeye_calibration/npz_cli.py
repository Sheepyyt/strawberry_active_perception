"""Disk-only CLI for converting capture NPZs to the hand-eye core JSON schema."""

from __future__ import annotations

import argparse
import json

from .npz_session import load_capture_session
from .schema import save_dataset


def main() -> None:
    """Validate and convert saved NPZ captures without importing ROS."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate read-only hand-eye capture NPZs and write one versioned "
            "offline solver dataset. This command never contacts the robot."
        )
    )
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        help="optional immutable subset, for example pose_001 pose_002",
    )
    arguments = parser.parse_args()
    dataset = load_capture_session(
        arguments.input_directory,
        sample_ids=arguments.sample_ids,
    )
    save_dataset(dataset, arguments.output)
    print(
        json.dumps(
            {
                "status": "converted",
                "session_id": dataset.session_id,
                "sample_count": len(dataset.samples),
                "sample_ids": [sample.sample_id for sample in dataset.samples],
                "output": arguments.output,
                "robot_contacted": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

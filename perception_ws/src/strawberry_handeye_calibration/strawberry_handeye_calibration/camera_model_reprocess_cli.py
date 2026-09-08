"""CLI for strict, disk-only camera-model reprocessing of capture NPZs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .camera_model_reprocess import reprocess_capture_session
from .schema import save_dataset


def main() -> None:
    """Recompute checkerboard PnP without importing ROS or touching hardware."""
    parser = argparse.ArgumentParser(
        description=(
            "Recompute saved T_camera_checkerboard poses with an explicit "
            "versioned camera model and cv2 SOLVEPNP_IPPE. Source NPZ files "
            "remain unchanged; this command never contacts the robot."
        )
    )
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--camera-model", required=True)
    parser.add_argument("--output", required=True, help="new CalibrationDataset v1 JSON")
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        help="optional immutable subset, for example pose_001 pose_002",
    )
    parser.add_argument(
        "--allow-k-change",
        action="store_true",
        help=(
            "diagnostic-only escape hatch; default operation requires model K "
            "to equal the capture K"
        ),
    )
    arguments = parser.parse_args()
    model_path = Path(arguments.camera_model).expanduser().resolve()
    output_path = Path(arguments.output).expanduser().resolve()
    if output_path.suffix.lower() != ".json":
        parser.error("--output must name a .json file; source NPZs are never overwritten")
    if output_path == model_path:
        parser.error("--output must not overwrite the input camera-model JSON")
    dataset = reprocess_capture_session(
        arguments.input_directory,
        model_path,
        sample_ids=arguments.sample_ids,
        allow_k_change=arguments.allow_k_change,
    )
    save_dataset(dataset, output_path)
    audit = dataset.metadata["camera_model_reprocessing"]
    print(
        json.dumps(
            {
                "status": "reprocessed",
                "session_id": dataset.session_id,
                "sample_count": len(dataset.samples),
                "camera_model_sha256": audit["camera_model_sha256"],
                "K_changed_beyond_tolerance": audit["K_policy"][
                    "K_changed_beyond_tolerance"
                ],
                "safe_for_robot_use": False,
                "output": str(output_path),
                "robot_contacted": False,
                "source_npz_modified": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

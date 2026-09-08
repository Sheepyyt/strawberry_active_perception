"""NPZ fixture replay helpers and a small ROS-free command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .fixtures import (
    FixtureObservation,
    ObservationFixture,
    load_fixture_npz,
    make_multiview_fixture,
    make_plane_fixture,
    save_fixture_npz,
)


class NPZReplay:
    """Replay immutable observations in deterministic timestamp order."""

    def __init__(self, path: str | Path):
        self.fixture = load_fixture_npz(path)

    def __iter__(self) -> Iterator[FixtureObservation]:
        return self.fixture.observations()

    def __len__(self) -> int:
        return len(self.fixture)


def replay_npz(path: str | Path) -> Iterator[FixtureObservation]:
    """Yield observations for offline and ROS topic adapters."""
    return iter(NPZReplay(path))


def fixture_summary(fixture: ObservationFixture) -> dict[str, object]:
    finite = np.isfinite(fixture.depth)
    return {
        "scene_id": fixture.scene_id,
        "observations": len(fixture),
        "shape": list(fixture.depth.shape[1:]),
        "valid_depth_fraction": float(np.count_nonzero(finite) / finite.size),
        "masked_pixels": [int(np.count_nonzero(view)) for view in fixture.mask],
        "first_stamp": float(fixture.stamp[0]),
        "last_stamp": float(fixture.stamp[-1]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="write a deterministic fixture NPZ")
    generate.add_argument("kind", choices=("plane", "multiview"))
    generate.add_argument("output", type=Path)
    generate.add_argument("--width", type=int, default=160)
    generate.add_argument("--height", type=int, default=100)
    generate.add_argument("--distance", type=float, default=1.0, help="plane distance in metres")
    inspect = subparsers.add_parser("inspect", help="validate and summarize an NPZ")
    inspect.add_argument("path", type=Path)
    replay = subparsers.add_parser("replay", help="emit one JSON record per observation")
    replay.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        fixture = (
            make_plane_fixture(args.width, args.height, args.distance)
            if args.kind == "plane"
            else make_multiview_fixture(args.width, args.height)
        )
        save_fixture_npz(fixture, args.output)
        print(json.dumps(fixture_summary(fixture), sort_keys=True))
        return 0
    fixture = load_fixture_npz(args.path)
    if args.command == "inspect":
        print(json.dumps(fixture_summary(fixture), sort_keys=True))
        return 0
    for observation in fixture.observations():
        record = {
            "scene_id": observation.scene_id,
            "observation_id": observation.observation_id,
            "stamp": observation.stamp,
            "shape": list(observation.depth.shape),
            "valid_depth_fraction": float(
                np.count_nonzero(np.isfinite(observation.depth))
                / observation.depth.size
            ),
            "masked_pixels": int(np.count_nonzero(observation.mask)),
        }
        print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

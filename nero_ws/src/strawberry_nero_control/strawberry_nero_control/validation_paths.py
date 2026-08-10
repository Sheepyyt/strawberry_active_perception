"""Project and runtime paths for durable NERO validation evidence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT_ENV = "STRAWBERRY_ACTIVE_PERCEPTION_ROOT"


def _looks_like_project_root(path: Path) -> bool:
    return (
        (path / ".git").exists()
        and (
            path
            / "nero_ws"
            / "src"
            / "strawberry_nero_control"
        ).is_dir()
    )


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the shared repository without depending on a hidden ROS path."""
    configured = os.environ.get(PROJECT_ROOT_ENV)
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not _looks_like_project_root(candidate):
            raise RuntimeError(
                f"{PROJECT_ROOT_ENV} does not point to this project: "
                f"{candidate}"
            )
        return candidate

    seeds = [
        Path.cwd().resolve(),
        Path(__file__).resolve(),
    ]
    visited = set()
    for seed in seeds:
        candidates = (seed, *seed.parents)
        for candidate in candidates:
            if candidate in visited:
                continue
            visited.add(candidate)
            if _looks_like_project_root(candidate):
                return candidate
    raise RuntimeError(
        "cannot find strawberry_active_perception project root; run from the "
        f"repository or set {PROJECT_ROOT_ENV}"
    )


def week1_validation_directory(
    override: Optional[Path] = None,
) -> Path:
    """Return the visible, Git-trackable Week-1 evidence directory."""
    if override is not None:
        return Path(override).expanduser().resolve()
    return find_project_root() / "validation" / "week1"


def week1_runtime_directory(
    override: Optional[Path] = None,
) -> Path:
    """Return the hidden directory for resumable intermediate sessions."""
    if override is not None:
        return Path(override).expanduser().resolve()
    return (
        Path.home()
        / ".ros"
        / "strawberry_nero_control"
        / "week1_acceptance_sessions"
    )

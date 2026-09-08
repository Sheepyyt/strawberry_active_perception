"""Run the same flake8 policy as the existing NERO control package."""

from pathlib import Path

from ament_flake8.main import main_with_errors
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8() -> None:
    """Reject actionable source-style errors while retaining project style."""
    return_code, errors = main_with_errors(
        argv=[
            "--config",
            str(PACKAGE_ROOT / "setup.cfg"),
            str(PACKAGE_ROOT / "launch"),
            str(PACKAGE_ROOT / "setup.py"),
            str(PACKAGE_ROOT / "strawberry_active_perception_bridge"),
            str(PACKAGE_ROOT / "test"),
        ]
    )
    assert return_code == 0, "\n".join(errors)

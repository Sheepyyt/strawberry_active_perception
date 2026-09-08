"""Run the repository's ROS Python flake8 policy."""

from pathlib import Path

from ament_flake8.main import main_with_errors
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8() -> None:
    """Reject actionable Python source-style errors."""
    return_code, errors = main_with_errors(
        argv=[
            "--config",
            str(PACKAGE_ROOT / "setup.cfg"),
            str(PACKAGE_ROOT / "launch"),
            str(PACKAGE_ROOT / "setup.py"),
            str(PACKAGE_ROOT / "strawberry_gradient_nbv"),
            str(PACKAGE_ROOT / "test"),
        ]
    )
    assert return_code == 0, "\n".join(errors)

"""Tests for real-arm launch safety gates."""

import importlib.util
from pathlib import Path

from launch import LaunchContext
import pytest


LAUNCH_PATH = Path(__file__).resolve().parents[1] / 'launch' / 'real.launch.py'


def _load_real_launch():
    spec = importlib.util.spec_from_file_location('nero_real_launch', LAUNCH_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context(**overrides):
    values = {
        'allow_limit_recovery_execution': 'false',
        'speed_percent': '0',
        'first_motion_test_mode': 'false',
        'precision_test_mode': 'false',
        'startup_enable': 'false',
    }
    values.update(overrides)
    context = LaunchContext()
    context.launch_configurations.update(values)
    return context


def test_default_real_launch_stays_disabled_and_read_only():
    module = _load_real_launch()

    assert module._validate_supervised_options(_context()) == []


def test_startup_enable_requires_ten_percent_and_supervised_mode():
    module = _load_real_launch()

    with pytest.raises(RuntimeError, match='speed_percent=10'):
        module._validate_supervised_options(_context(startup_enable='true'))

    with pytest.raises(RuntimeError, match='explicit supervised'):
        module._validate_supervised_options(_context(
            startup_enable='true',
            speed_percent='10',
        ))


def test_startup_enable_is_allowed_for_precision_test_at_ten_percent():
    module = _load_real_launch()
    context = _context(
        startup_enable='true',
        speed_percent='10',
        precision_test_mode='true',
    )

    assert module._validate_supervised_options(context) == []

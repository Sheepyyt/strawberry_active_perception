"""Offline authorization and evidence checks for the existing recovery Action."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("recover_start", ROOT / "tools/demo/recover_start.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)
CODES = SimpleNamespace(SUCCESS=0, ALREADY_SAFE=1, PREVIEW_READY=2)


def result(code, *, success=True, executed=False, safe=False):
    return SimpleNamespace(code=code, success=success, executed=executed, robot_is_safe=safe)


class FakeNode:
    def __init__(self, previews, execution=None):
        self.previews = iter(previews)
        self.execution = execution
        self.execute_calls = 0

    def preview_recovery(self):
        return next(self.previews)

    def execute_recovery(self):
        self.execute_calls += 1
        return self.execution


def arguments(tmp_path, **changes):
    return SimpleNamespace(**{"output_dir": tmp_path, "execute": True, "workspace_clear": True, **changes})


def test_already_safe_does_not_request_motion(tmp_path):
    node = FakeNode([result(CODES.ALREADY_SAFE, safe=True)])
    report = recovery.recover(node, arguments(tmp_path), CODES, vars)
    assert report == {"status": "already_safe", "executed": False, "robot_is_safe": True}
    assert node.execute_calls == 0
    assert json.loads((tmp_path / "preview.json").read_text())["robot_is_safe"] is True


def test_preview_only_never_requests_motion(tmp_path):
    node = FakeNode([result(CODES.PREVIEW_READY)])
    report = recovery.recover(node, arguments(tmp_path, execute=False), CODES, vars)
    assert report["status"] == "preview_ready"
    assert node.execute_calls == 0


@pytest.mark.parametrize("preview", [result(21, success=False), result(CODES.PREVIEW_READY, safe=True)])
def test_failed_or_contradictory_preview_never_executes(tmp_path, preview):
    node = FakeNode([preview])
    with pytest.raises(RuntimeError, match="恢复预览未通过"):
        recovery.recover(node, arguments(tmp_path), CODES, vars)
    assert node.execute_calls == 0


def test_motion_requires_workspace_clearance(tmp_path):
    node = FakeNode([])
    with pytest.raises(RuntimeError, match="workspace-clear"):
        recovery.recover(node, arguments(tmp_path, workspace_clear=False), CODES, vars)
    assert node.execute_calls == 0


def test_recovery_requires_fresh_already_safe_confirmation(tmp_path):
    node = FakeNode([result(CODES.PREVIEW_READY), result(CODES.ALREADY_SAFE, safe=True)],
                    result(CODES.SUCCESS, executed=True, safe=True))
    report = recovery.recover(node, arguments(tmp_path), CODES, vars)
    assert report["status"] == "recovered"
    assert node.execute_calls == 1
    assert {path.name for path in tmp_path.iterdir()} == {"preview.json", "execution.json", "post_preview.json"}


def test_abort_after_preview_prevents_motion(tmp_path):
    abort_file = tmp_path / "abort"
    args = arguments(tmp_path, abort_file=abort_file, abort_not_before_ns=0)
    node = FakeNode([])

    def preview():
        abort_file.touch()
        return result(CODES.PREVIEW_READY)

    node.preview_recovery = preview
    with pytest.raises(RuntimeError, match="已请求停止"):
        recovery.recover(node, args, CODES, vars)
    assert node.execute_calls == 0

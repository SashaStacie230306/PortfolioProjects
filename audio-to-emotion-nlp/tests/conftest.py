"""Conftest file for tests."""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SPHINX_MOCK_MODE", "1")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolate_fs(tmp_path, monkeypatch, request):
    """Isolate the filesystem for tests, except for mp3‐asset test.

    By default, switch into an empty tmp_path, but *don’t* isolate the filesystem for
    the one mp3‐asset test.
    """
    asset_test_name = "test_sondy_fixed_mp3_exists_and_nonempty"
    if asset_test_name in request.node.name:
        # Let that test see the real files in the repo
        return
    # Otherwise sandbox
    monkeypatch.chdir(tmp_path)

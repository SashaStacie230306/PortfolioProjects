#!/usr/bin/env python3
"""test_main.py - Minimal coverage tests for src.cli.main."""

import os

import pytest
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()


@pytest.fixture(scope="module", autouse=True)
def enable_mock_mode():
    """Enable mock mode for all CLI tests."""
    os.environ["SPHINX_MOCK_MODE"] = "1"
    yield
    os.environ.pop("SPHINX_MOCK_MODE", None)


def test_text_command():
    """Test the 'text' command."""
    result = runner.invoke(app, ["text", "I am happy!"])
    assert result.exit_code == 0
    assert "[MOCK]" in result.stdout


def test_audio_command(tmp_path):
    """Test the 'audio' command with a dummy file."""
    audio_file = tmp_path / "dummy.mp3"
    audio_file.write_text("dummy content")
    result = runner.invoke(app, ["audio", str(audio_file)])
    assert result.exit_code == 0
    assert "[MOCK]" in result.stdout


def test_video_command(tmp_path):
    """Test the 'video' command with a dummy file."""
    video_file = tmp_path / "dummy.mp4"
    video_file.write_text("dummy content")
    result = runner.invoke(app, ["video", str(video_file)])
    assert result.exit_code == 0
    assert "[MOCK]" in result.stdout


def test_url_command():
    """Test the 'url' command."""
    result = runner.invoke(app, ["url", "https://example.com/fake.mp4"])
    assert result.exit_code == 0
    assert "[MOCK]" in result.stdout

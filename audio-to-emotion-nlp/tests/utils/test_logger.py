"""Unit tests for src/utils/logger.py (console-only, no filesystem I/O)."""

import logging
from unittest.mock import patch

import pytest

import utils.logger as logger_mod


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    """Ensure a clean test environment."""
    monkeypatch.chdir(tmp_path)
    for handler in list(logging.root.handlers):
        logging.root.removeHandler(handler)
    yield
    for handler in list(logging.root.handlers):
        logging.root.removeHandler(handler)


def test_setup_logging_console_only(tmp_path, caplog):
    """Test setup_logging creates console output without writing to a file."""
    caplog.set_level(logging.INFO)

    with patch("utils.logger.Path.mkdir"), patch(
        "utils.logger.Path.__truediv__", return_value="mocked.log"
    ), patch("logging.config.dictConfig") as mock_config:
        logger_mod.setup_logging(log_dir=str(tmp_path / "logs"))

    assert mock_config.called, "Logging configuration not initialized"

    log = logger_mod.get_logger("test_logger")
    log.info("console-only test")
    assert "console-only test" in [rec.message for rec in caplog.records]


def test_get_logger_singleton():
    """get_logger returns the same Logger instance by name."""
    lg1 = logger_mod.get_logger("foo")
    lg2 = logger_mod.get_logger("foo")
    assert lg1 is lg2
    assert lg1.name == "foo"


def test_setup_logging_idempotent():
    """Calling setup_logging multiple times doesn't add more handlers."""
    logger_mod.setup_logging()
    count1 = len(logging.root.handlers)
    logger_mod.setup_logging()
    count2 = len(logging.root.handlers)
    assert count2 == count1

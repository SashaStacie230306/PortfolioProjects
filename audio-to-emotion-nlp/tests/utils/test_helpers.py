"""Test actual functions in utils/helpers.py to boost coverage."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_setup_logging():
    """Test setup_logging function - lines 19-26."""
    from utils.helpers import logger, setup_logging

    # Remove actual handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    with patch("os.makedirs") as mock_makedirs, patch(
        "logging.basicConfig"
    ) as mock_basic_config, patch(
        "logging.FileHandler"
    ) as mock_file_handler, patch.object(
        logger, "hasHandlers", return_value=False
    ):
        setup_logging()
        mock_makedirs.assert_called_once()
        mock_basic_config.assert_called_once()
        mock_file_handler.assert_called_once()


def test_log_message_all_levels():
    """Test log_message function with all levels - lines 43-50."""
    from utils.helpers import log_message

    with patch("utils.helpers.logger") as mock_logger:
        # Test all valid levels
        log_message("info", "Test info message")
        mock_logger.info.assert_called_with("Test info message")

        log_message("error", "Test error message")
        mock_logger.error.assert_called_with("Test error message")

        log_message("debug", "Test debug message")
        mock_logger.debug.assert_called_with("Test debug message")

        # Test unknown level (should trigger warning)
        log_message("unknown", "Test unknown message")
        mock_logger.warning.assert_called_with(
            "Unknown log level %r, message=%r", "unknown", "Test unknown message"
        )


def test_save_to_csv_comprehensive():
    """Test save_to_csv function comprehensively - lines 80-104."""
    from utils.helpers import save_to_csv

    # Create test dataframe
    test_df = pd.DataFrame(
        {
            "text": ["hello", "world"],
            "emotion": ["happy", "sad"],
            "confidence": [0.9, 0.8],
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: New file (file doesn't exist)
        with patch("os.makedirs") as mock_makedirs:
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                with patch("os.path.exists", return_value=False):
                    save_to_csv(test_df, "test.csv", temp_dir)
                    mock_makedirs.assert_called_once()
                    mock_to_csv.assert_called_once()

        # Test 2: File exists - successful merge
        existing_df = pd.DataFrame(
            {"text": ["existing"], "emotion": ["neutral"], "confidence": [0.5]}
        )

        with patch("os.path.exists", return_value=True):
            with patch("pandas.read_csv", return_value=existing_df):
                with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                    save_to_csv(test_df, "test.csv", temp_dir)
                    mock_to_csv.assert_called_once()

        # Test 3: File exists but read fails (exception handling)
        with patch("os.path.exists", return_value=True):
            with patch("pandas.read_csv", side_effect=Exception("Read failed")):
                with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                    save_to_csv(test_df, "test.csv", temp_dir)
                    mock_to_csv.assert_called_once()

        # Test 4: Default output_dir (None)
        with patch("os.makedirs") as mock_makedirs:
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                with patch("os.path.exists", return_value=False):
                    save_to_csv(test_df, "test.csv", None)  # output_dir=None
                    mock_makedirs.assert_called_once()
                    mock_to_csv.assert_called_once()


def test_edge_cases():
    """Test edge cases and error conditions."""
    from utils.helpers import format_time, log_message, save_to_csv

    # Test format_time with edge cases
    assert format_time(0) == "00:00:00,000"
    assert format_time(999) == "00:00:00,999"

    # Test log_message with empty message
    with patch("utils.helpers.logger") as mock_logger:
        log_message("info", "")
        mock_logger.info.assert_called_with("")

    # Test save_to_csv with empty dataframe
    empty_df = pd.DataFrame()
    with patch("pandas.DataFrame.to_csv") as mock_to_csv:
        with patch("os.path.exists", return_value=False):
            save_to_csv(empty_df, "empty.csv", "temp")
            mock_to_csv.assert_called_once()


def test_logging_integration():
    """Test logging setup integration."""
    # Clear handlers
    from utils.helpers import log_message, logger, setup_logging

    logger.handlers.clear()

    # Test complete logging flow
    with patch("os.makedirs"):
        with patch("logging.basicConfig"):
            setup_logging()

            # Test that logging works after setup
            with patch.object(logger, "info") as mock_info:
                log_message("info", "Integration test")
                mock_info.assert_called_with("Integration test")


def test_save_to_csv_different_scenarios():
    """Test save_to_csv with different scenarios to increase coverage."""
    from utils.helpers import save_to_csv

    # Test with different dataframe structures
    test_cases = [
        pd.DataFrame({"single_col": [1, 2, 3]}),
        pd.DataFrame({"col1": ["a"], "col2": ["b"], "col3": ["c"]}),
        pd.DataFrame({"numbers": [1.1, 2.2, 3.3], "strings": ["x", "y", "z"]}),
    ]

    for i, df in enumerate(test_cases):
        filename = f"test_{i}.csv"

        # Test new file creation
        with patch("os.path.exists", return_value=False):
            with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                with patch("os.makedirs"):
                    save_to_csv(df, filename, "test_dir")
                    mock_to_csv.assert_called_once()

        # Test file merging
        with patch("os.path.exists", return_value=True):
            with patch("pandas.read_csv", return_value=df):
                with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                    save_to_csv(df, filename, "test_dir")
                    mock_to_csv.assert_called_once()

"""Utility functions for logging, time formatting, and CSV handling."""

import logging
import os
from typing import Optional

import pandas as pd

# Module‐scoped logger
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging to output messages to both the terminal and a log file.

    Creates a 'src/logs' directory if it doesn't exist.  Console shows INFO+, file
    captures DEBUG+ in 'src/logs/app.log'.
    """
    log_dir = os.path.join("src", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # If handlers are already configured, do nothing
    if logger.hasHandlers():
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "app.log")),
        ],
    )


def log_message(level: str, message: str) -> None:
    """Log a message at a specified severity level.

    Args:
        level: One of "info", "error", "debug".
        message: The log text.
    """
    if level == "info":
        logger.info(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        logger.debug(message)
    else:
        logger.warning("Unknown log level %r, message=%r", level, message)


def format_time(ms: int) -> str:
    """Convert milliseconds to a subtitle‐style timestamp "HH:MM:SS,mmm"."""
    seconds = ms / 1000.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{millis:03}"


def save_to_csv(
    df: pd.DataFrame,
    filename: str = "final_output.csv",
    output_dir: Optional[str] = None,
) -> None:
    """Save a DataFrame to CSV, prepending new rows above existing ones.

    If the file already exists:
      - read it in
      - new rows go on top
      - old rows get pushed down

    Args:
      df: DataFrame of new rows
      filename: CSV filename
      output_dir: base directory (defaults to 'src/data/processed')
    """
    if output_dir is None:
        output_dir = os.path.join("src", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)

    if os.path.exists(file_path):
        try:
            old_df = pd.read_csv(file_path)
            combined = pd.concat([df, old_df], ignore_index=True)
            logger.debug(
                "Prepending %d new rows to existing %d rows in %s",
                len(df),
                len(old_df),
                file_path,
            )
        except Exception as e:
            logger.error(
                "Failed to read existing CSV %s: %s; overwriting", file_path, e
            )
            combined = df
    else:
        combined = df
    combined.to_csv(file_path, index=False)
    logger.info("Saved output to %s (total %d rows)", file_path, len(combined))

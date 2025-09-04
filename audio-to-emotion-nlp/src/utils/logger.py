"""Logging utility for the application."""

import logging
import logging.config
from pathlib import Path


def setup_logging(log_dir: str = "src/logs", log_filename: str = "app.log") -> None:
    """Configure application-wide logging.

    • Console handler at INFO+
    • File handler capturing DEBUG+
    • Root logger at DEBUG
    • urllib3 logger elevated to INFO to suppress its DEBUGs

    Args:
        log_dir: Directory under which to write the log file.
        log_filename: Name of the log file.
    """
    # Ensure the log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logfile_path = Path(log_dir) / log_filename

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": str(logfile_path),
                "mode": "a",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": False,
            },
            "urllib3": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Retrieve a named logger.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)

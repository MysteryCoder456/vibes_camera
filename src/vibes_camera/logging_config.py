"""Logging configuration for vibes_camera.

Sets up logging with both console and file handlers.
Logs are saved daily to ~/.config/vibes_camera/logs/YYYY-MM-DD.log

File handler: Logs at specified level (default INFO) - captures all events
Console handler: Only WARNING and above - keeps terminal clean
"""

import logging
from datetime import date
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "vibes_camera"
LOGS_DIR = CONFIG_DIR / "logs"

# Custom formatter with 12-hour AM/PM time
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %I:%M:%S.%f %p"


class MillisecondFormatter(logging.Formatter):
    """Custom formatter that includes milliseconds in 12-hour AM/PM format."""

    def formatTime(self, record, datefmt=None):
        """Format time with milliseconds and 12-hour AM/PM."""
        from datetime import datetime

        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            # Replace %f with actual milliseconds (3 digits)
            s = ct.strftime(datefmt.replace("%f", f"{int(ct.microsecond / 1000):03d}"))
        else:
            s = ct.strftime("%Y-%m-%d %I:%M:%S.%f %p")
        return s


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging with console and file handlers.

    Args:
        level: Logging level for file output (e.g., logging.DEBUG, logging.INFO)
               Console always shows WARNING and above only.

    Returns:
        Configured logger instance
    """
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Daily log file
    log_file = LOGS_DIR / f"{date.today().isoformat()}.log"

    # Get or create logger
    logger = logging.getLogger("vibes_camera")
    logger.setLevel(level)

    # Clear existing handlers (in case setup_logging is called multiple times)
    logger.handlers.clear()

    # Create formatter
    formatter = MillisecondFormatter(LOG_FORMAT, DATE_FORMAT)

    # File handler - logs at specified level (default INFO)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler - only WARNING and above to keep terminal clean
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger() -> logging.Logger:
    """Get the vibes_camera logger.

    Returns:
        The vibes_camera logger instance
    """
    return logging.getLogger("vibes_camera")

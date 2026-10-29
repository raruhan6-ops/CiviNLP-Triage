"""
Centralized logging configuration for CiviNLP Triage.

Provides structured logging to both console and timestamped log files.
Import and call setup_logging() at the entry point of any script.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    name: str = "civinlp",
) -> logging.Logger:
    """
    Configure logging with console and file handlers.

    Args:
        log_dir: Directory to store log files.
        level: Logging level (default INFO).
        name: Logger name / log file prefix.

    Returns:
        Configured root logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Replace handlers from an earlier setup_logging call; leave others (e.g.
    # pytest's) alone. Only add a console handler if none writes to a stream yet.
    for handler in list(root_logger.handlers):
        if getattr(handler, "_civinlp", False):
            root_logger.removeHandler(handler)
            handler.close()
    has_console = any(type(h) is logging.StreamHandler for h in root_logger.handlers)
    for handler in ([file_handler] if has_console else [console_handler, file_handler]):
        handler._civinlp = True
        root_logger.addHandler(handler)

    logging.getLogger(__name__).info(f"Logging initialized. Log file: {log_file}")
    return root_logger

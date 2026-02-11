"""Logging utilities for VisionLLM InteractionAnalysis."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_LOGS_DIR = os.environ.get("VISIONLLM_LOGS_DIR", "logs")
Path(_LOGS_DIR).mkdir(parents=True, exist_ok=True)

_LOG_FILE = os.path.join(_LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_formatter = logging.Formatter(_FMT)

_file_handler = logging.FileHandler(_LOG_FILE)
_file_handler.setFormatter(_formatter)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with file + console handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        logger.addHandler(_file_handler)
        logger.addHandler(_console_handler)
    return logger

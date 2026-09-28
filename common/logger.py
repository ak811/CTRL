"""Uniform console + file logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly (no duplicate handlers)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        logger.addHandler(stream)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
            for h in logger.handlers
        ):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
            logger.addHandler(fh)
    return logger

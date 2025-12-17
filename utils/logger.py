"""Logging utility for AgentForge."""
import logging
import os
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Optional log level override

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Set level from env or default
        log_level = level or os.getenv("LOG_LEVEL", "INFO")
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Console handler
        handler = logging.StreamHandler()
        handler.setLevel(logger.level)

        # Format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

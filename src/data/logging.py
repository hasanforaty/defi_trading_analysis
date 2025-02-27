# src/data/logging.py
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

from loguru import logger
from config.settings import LoggingSettings, get_settings


class InterceptHandler(logging.Handler):
    """
    Intercepts standard library logging and redirects to loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """
    Configure logging using loguru with both console and file handlers.
    Intercepts standard library logging.
    """
    # Get logging settings
    settings = get_settings()
    log_settings = settings.logging

    # Remove default logger
    logger.remove()

    # Ensure log directory exists
    log_file_path = Path(log_settings.log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Add console handler
    logger.add(
        sys.stderr,
        level=log_settings.level.value,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Add file handler if enabled
    if log_settings.log_to_file:
        logger.add(
            log_settings.log_file_path,
            level=log_settings.level.value,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=log_settings.log_rotation,
            retention=log_settings.log_retention,
            compression=log_settings.log_compression,
        )

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Intercept uvicorn, fastapi, and other libraries that use standard logging
    for logger_name in ("uvicorn", "uvicorn.access", "fastapi", "sqlalchemy.engine"):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]

    logger.info(f"Logging initialized at level {log_settings.level.value}")

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from typing import Optional

from audio_analyzer.core.settings import settings


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger for the given name.
    
    If the logger has no handlers, attach a StreamHandler that writes to stdout, set the logger level based on settings.DEBUG, and apply a formatter that includes timestamp, logger name, filename, line number, level, and message.
    
    Parameters:
        name (Optional[str]): Logger name; defaults to the module's __name__ when omitted.
    
    Returns:
        logging.Logger: The configured logger instance. The function avoids adding duplicate handlers to an existing logger.
    """
    logger_name = name if name else __name__
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        # Set log level based on debug setting
        log_level = logging.DEBUG if settings.DEBUG else logging.INFO
        logger.setLevel(log_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        # Create formatter with filename and line number
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


logger = setup_logger("audio_analyzer")

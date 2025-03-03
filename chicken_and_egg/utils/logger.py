"""
Setup logger
"""

import sys

from loguru import logger

logger_format = "<level>{message}</level>"
logger.remove()
logger.add(sys.stderr, format=logger_format)


def log(message: str, color: str = ""):
    if color:
        logger.opt(colors=True).info(f"<{color}>{message}</{color}>")
    else:
        logger.info(message)

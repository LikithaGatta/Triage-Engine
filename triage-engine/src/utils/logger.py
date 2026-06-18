

import sys
from loguru import logger

# Remove the default loguru handler so we can add our own
logger.remove()

# Add a colored terminal handler
# {time} = timestamp, {level} = INFO/WARNING/ERROR, {message} = your text
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

# Add a file handler — saves logs to a file for later review
# rotation="10 MB" starts a new file when the current one hits 10MB
logger.add(
    "logs/triage.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="10 MB",
)

# __all__ controls what gets exported when another file does:
# from src.utils.logger import *
__all__ = ["logger"]
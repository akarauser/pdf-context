import logging
import os
from pathlib import Path


class InfoFilter(logging.Filter):
    """
    Info messages filter for logging handlers.
    """

    def filter(self, record) -> bool:
        return record.levelno == logging.INFO


class NonInfoFilter(logging.Filter):
    """
    Non Info messages filter for logging handlers.
    """

    def filter(self, record) -> bool:
        return record.levelno != logging.INFO


class LoggingConfig:
    """Logging handler setups."""

    @staticmethod
    def _stream_handler_config() -> None:
        """Initialize StreanHandler"""
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter(
            "%(levelname)s %(pathname)s:%(funcName)s:%(lineno)d -> %(asctime)s %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        stream_handler.setFormatter(stream_formatter)
        stream_handler.addFilter(NonInfoFilter())
        logger.addHandler(stream_handler)

    @staticmethod
    def _file_handler_config() -> None:
        """Initialize FileHandler"""
        try:
            logs_folder = Path(__file__).parents[1] / "logs"
            if not logs_folder.exists():
                os.makedirs(logs_folder)

            file_handler = logging.FileHandler(f"{logs_folder}\\info.log")
            file_formatter = logging.Formatter(
                "%(asctime)s %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(InfoFilter())
            logger.addHandler(file_handler)
        except FileNotFoundError:
            raise


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LoggingConfig._stream_handler_config()
# LoggingConfig._file_handler_config()

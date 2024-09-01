"""
This module manages logger configuration.

LoggerSettings
"""

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggerSettings(BaseSettings):
    """
    A class that manages the logger settings.

    This class manages the logger settings.
    These environment variable log_level is loaded from an .env file.

    Attributes:
        log_level: Base level used for logging.
    """

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    log_level: str


def configure_logging(log_level: str) -> None:
    """
    Configure the logging.

    Args:
        log_level (str): Logging level for the logger
    """
    logger.remove()
    logger.add(
        'logs/app.log',
        rotation='1 day',
        retention='2 days',
        compression='zip',
        level=log_level,
    )


# initialize logger
configure_logging(log_level=LoggerSettings().log_level)

"""
This module manages the ML model configuration.

The ModelSettings class contains the various settings and hyperparameters
relevant to the ML model. This is done through Pydantic's BaseSettings.
"""

from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """
    ML model that manages the application settings.

    This class manages the application settings used throughout
    the src folder. These environment variables are loaded from a .env file.

    Attributes:
        ml_model_path: Directory containing the ML model.
        ml_model_name: Name of the ML model file.
    """

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    ml_model_path: DirectoryPath
    ml_model_name: str


# initialize model settings
model_settings = ModelSettings()

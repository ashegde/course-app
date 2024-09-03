"""
This module manages model loading and building.

The ModelBuilderService class handles the building of an ML model.
"""

from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelBuilderService:
    """
    A class that builds an ML model.

    This class provides functionality for building and saving an ML model.

    Attributes:
        model_path (str): path to model directory
        model_name (str): name of the model

    Methods:
        __init__: Constructor that initializes the ModelBuilderService object.
        train_model: Trains an ML model.
    """

    def __init__(self) -> None:
        """Initialize ModelBuilderService with no model loaded."""
        self.model_path = model_settings.ml_model_path
        self.model_name = model_settings.ml_model_name

    def train_model(self) -> None:
        """Trains a new model and saves it to the dir in settings."""
        logger.warning(
            'building a new model at ',
            f'{self.model_path}/{self.model_name}',
        )
        build_model()

"""
This module manages model loading and building.

The ModelInferenceService class handles predictions with an ML model.
"""

import pickle as pkl
from pathlib import Path

from loguru import logger

from config import model_settings


class ModelInferenceService:
    """
    A class that manages an ML model.

    This class provides functionality for loading an ML model and making
    predictions.

    Attributes:
        model: ML model that is managed by this service. Initialized as None.
        model_path (str): path to model directory
        model_name (str): name of the model

    Methods:
        __init__: Constructor that initializes the ModelService object.
        load_model: Loads the model from a file, or builds a new model.
        predict: Predict with the loaded model.
    """

    def __init__(self) -> None:
        """Initialize the served model to None."""
        self.model = None
        self.model_path = model_settings.ml_model_path
        self.model_name = model_settings.ml_model_name

    def load_model(self) -> None:
        """
        Load the model at a specified path.

        Raises:
            FileNotFoundError: if the specified path does not exist
        """
        model_path = Path(
            f'{self.model_path}/{self.model_name}',
        )
        logger.info(f'verifying existence of model file at {model_path}')

        if not model_path.exists():
            raise FileNotFoundError(
                f'Model at {self.model_path}/{self.model_name} does not exist',
            )

        with open(model_path, 'rb') as model_file:
            self.model = pkl.load(model_file)

    def predict(self, input_features: list[float]) -> list[float]:
        """
        Make predictions with the stored model.

        Makes predictions using the loaded model at the specified inputs.

        Args:
            input_features (list): input data for the prediction.

        Returns:
            list: The prediction output by the model.

        Raises:
            FileNotFoundError: if no model is loaded
        """
        logger.info('making prediction')
        if self.model is None:
            raise FileNotFoundError('No model loaded.')
        return self.model.predict([input_features])

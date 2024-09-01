"""
This module manages model loading and prediction.

The ModelService class handles the loading an ML model if it exists.
If the model doesn't exist, then it is constructed.
This class also serves model predictions.
"""

import pickle as pkl
from pathlib import Path

from loguru import logger

from config import model_settings
from model.pipeline.model import build_model


class ModelService:
    """
    A class that manages an ML model.

    This class provides functionality for loading an ML model, if it exists,
    and building one if it does not, as well as making predictions.

    Attributes:
        model: ML model that is managed by this service. Initialized as None.

    Methods:
        __init__: Constructor that initializes the ModelService object.
        load_model: Loads the model from a file, or builds a new model.
        predict: Predict with the loaded model.
    """

    def __init__(self) -> None:
        """Initialize the served model to None."""
        self.model = None

    def load_model(self) -> None:
        """
        Load the model at a specified path if it exists.

        Otherwise builds a new model.
        """
        model_path = Path(
            f'{model_settings.ml_model_path}/{model_settings.ml_model_name}',
        )
        logger.info(f'verifying existence of model file at {model_path}')

        if not model_path.exists():
            logger.warning(
                f'{model_path} not found, '
                f'building a new model {model_settings.ml_model_name}',
            )
            build_model()

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
        """
        logger.info('making prediction')
        assert self.model is not None, 'no model loaded for prediction'
        return self.model.predict([input_features])

"""
This module provides functionality for running the application.

It illustrates how to load the ModelService and make predictions.
This can be run by directly entering `poetry run python runner.py`
or `python runner.py` into the terminal.
"""

from loguru import logger

from model.model_service import ModelService


@logger.catch
def main() -> None:
    """
    Load and run the ModelService.

    It loads the model and makes a prediction at a test input.
    The prediction is subsequently logged by the logger.
    """
    logger.info('running the application...')
    ml_svc = ModelService()
    ml_svc.load_model()
    input_features = {
        'area': 85,
        'construction_year': 2015,
        'bedrooms': 2,
        'garden': 20,
        'balcony_yes': 1,
        'parking_yes': 1,
        'furnished_yes': 0,
        'garage_yes': 0,
        'storage_yes': 1,
    }
    pred = ml_svc.predict(list(input_features.values()))
    logger.info(f'input: {input_features}\n -> prediction: {pred}')


if __name__ == '__main__':
    main()

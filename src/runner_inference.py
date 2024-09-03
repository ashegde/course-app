"""
This module provides functionality for performing inference.

It illustrates how to load the ModelInferenceService and make predictions
at pre-specified inputs. The resulting predictions are logged to the log file.
"""

from loguru import logger

from model.model_inference import ModelInferenceService


@logger.catch
def main() -> None:
    """
    Load and run the ModelService.

    It loads the model and makes a prediction at a test input.
    The prediction is subsequently logged by the logger.
    """
    logger.info('running the application...')
    ml_svc = ModelInferenceService()
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

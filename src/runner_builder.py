"""
This module provides functionality for running the ModelBuilderService.

It illustrates how to load the ModelBuilderService and train a model.
"""

from loguru import logger

from model.model_builder import ModelBuilderService


@logger.catch
def main() -> None:
    """
    Load and run the ModelBuilderService.

    It builds a new model.
    """
    logger.info('running the application...')
    build_svc = ModelBuilderService()
    build_svc.train_model()


if __name__ == '__main__':
    main()

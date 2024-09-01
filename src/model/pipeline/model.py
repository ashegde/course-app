"""
This module contains the methods for building an ML model.

Specifically, the methods below use `sklearn` to fit a RandomForestRegressor
to the dataset. This includes preprocessing steps to extract input and output
variables, split the dataset into train and test portions, etc.
"""

import os
import pickle as pkl

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from config import model_settings
from model.pipeline.preparation import prepare_data


def build_model() -> None:
    """
    Preprocesses data and fits an sklearn RandomForestRegressor model.

    This function extracts the data, splits it into training and test datasets,
    and then fits a RandomForestRegressor model. The model is saved to the
    directory specified in the settings configuration file.

    Args:
        None
    """
    logger.info('initiating model building pipeline')
    # 1. load data
    dataframe = prepare_data()
    # 2. get X, y dataset
    feature_names = [
        'area',
        'construction_year',
        'bedrooms',
        'garden',
        'balcony_yes',
        'parking_yes',
        'furnished_yes',
        'garage_yes',
        'storage_yes',
    ]
    x_data, y_data = _get_x_y(dataframe, feature_names)
    # 3. split the dataset
    x_train, x_test, y_train, y_test = _split_train_test(x_data, y_data)
    # 4. model training
    model = _train_model(x_train, y_train)
    # 5. evaluate the model
    _evaluate_model(model, x_test, y_test)
    # 6. save the model
    _save_model(model)


def _get_x_y(
    dataframe: pd.DataFrame,
    col_x: list[str],
    col_y: str = 'rent',
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract input and output variables from the DataFrame.

    Args:
        dataframe (pd.DataFrame): contains RentApartments data
        col_x (list[str]): List of column / feature names to extract as inputs
        col_y (str): Column / feature name to extract as output

    Returns:
        tuple[pd.DataFrame, pd.Series]: Extracted input and output variables
    """
    logger.info(
        'defining X and y variable.',
        f'\nX vars: {col_x}\ny vars: {col_y}',
    )

    return dataframe[col_x], dataframe[col_y]


def _split_train_test(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split the dataset into training and test portions.

    Performs an 80%-20% split of the dataset into training and test
    datasets.

    Args:
        features (pd.DataFrame): DataFrame containing input variables
        targets (pd.DataFrame): DataFrame containing output variables / targets
        test_size (float): fraction used for split

    Returns:
        list[pd.DataFrame]: containing x_train, y_train, x_test, y_test
    """
    logger.info('splitting x and y into training and test datasets.')
    return train_test_split(
        features,
        targets,
        test_size=test_size,
    )


def _train_model(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> RandomForestRegressor:
    """
    Trains an sklearn RandomForestRegressor model.

    Trains a RandomForestRegressor model using grid-search based
    cross validation to identift optimal model hyperparameters.

    Args:
        features (pd.DataFrame): DataFrame containing training inputs as rows
        targets (pd.DataFrame): DataFrame containing training outputs

    Returns:
        RandomForestRegressor: Best model learned through cross validation
    """
    logger.info('training RandomForestRegressor model')
    grid_space = {
        'n_estimators': [100, 200, 300],
        'max_depth': [2, 4, 8, 16, 32, 64],
    }
    logger.info(f'training RandomForestRegressor model with {grid_space}')
    grid = GridSearchCV(
        RandomForestRegressor(),
        param_grid=grid_space,
        cv=5,
        scoring='r2',
    )
    model_grid = grid.fit(features.values, targets.values)

    return model_grid.best_estimator_


def _evaluate_model(
    model: RandomForestRegressor,
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> float:
    """
    Evaluate the specfied model on the specified input data.

    Args:
        model (RandomForestRegressor): Model to be evaluated
        features (pd.DataFrame): input data
        targets (pd.DataFrame): output data, for scoring

    Returns:
        score (float): model score, compares model predictions to data
    """
    score = model.score(features.values, targets.values)
    logger.info(f'evaluating model performace. SCORE={score}')
    return score


def _save_model(model: RandomForestRegressor) -> None:
    """
    Save the model to the location specified in settings.

    Args:
        model (RandomForestRegressor): Model to be saved
    """
    os.makedirs(model_settings.ml_model_path, exist_ok=True)
    save_path = (
        f'{model_settings.ml_model_path}/{model_settings.ml_model_name}'
    )
    logger.info(f'saving model to {save_path}')
    with open(save_path, 'wb') as save_file:
        pkl.dump(model, save_file)

"""
This module contains functionality for preprocessing and preparing the data.

The functions below load the database and process relevant non-numerical
columns. The resulting DataFrame can be used for model training.
"""

import re

import pandas as pd
from loguru import logger

from model.pipeline.collection import load_data_from_db


def prepare_data() -> pd.DataFrame:
    """
    Prepare the data for model fitting.

    The data is loaded from the database and subsequently
    processed into a DataFrame suitable for model development.

    Args:
        None

    Returns:
        pd.DataFrame: Contains the prepared data
    """
    logger.info('initiating preprocessing pipeline')
    # 1. load dataset
    dataframe = load_data_from_db()
    # 2. encode categorical columns
    data_encoded = _encode_cat_cols(dataframe)
    # 3. parse the garden column
    return _parse_garden_col(data_encoded)


def _encode_cat_cols(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns of the RentApartments DataFrame.

    The columns ['balcony', 'parking', 'furnished', 'garage', 'storage']
    contain categorical values of 'yes' or 'no'. We use the pd.get_dummies
    to convert these columns to dummy/indicator variables.

    Args:
        dataframe (pd.DataFrame): RentApartments DataFrame to be processed

    Returns:
        pd.DataFrame: DataFrame with the above categorical columns encoded
    """
    cols = ['balcony', 'parking', 'furnished', 'garage', 'storage']
    logger.info(f'encoding categorical columns: {cols}')
    return pd.get_dummies(
        dataframe,
        columns=cols,
        drop_first=True,
    )


def _parse_garden_col(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the garden column of the RentApartments DataFrame.

    The garden column contains either the string 'Not present'
    or the string "Present (# m^2)". Below, we extract the
    numerical values from these strings.

    Args:
        dataframe (pd.DataFrame): DataFrame containing RentApartments data

    Returns:
        pd.DataFrame: DataFrame with a parsed 'garden' column.
    """
    logger.info('parsing the garden column')
    dataframe.garden = dataframe.garden.apply(
        lambda x: 0 if x == 'Not present' else int(re.findall(r'\d+', x)[0]),
    )
    return dataframe

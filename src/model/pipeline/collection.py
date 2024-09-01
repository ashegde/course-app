"""
This module contains functions for collecting or loading data from a database.

The primary function is `load_data_from_db`, which loads the sql database using
methods imported from SQLAlchemy. The read-in database is interpreted using the
RentApartments class and returned as a pandas DataFrame
"""

import pandas as pd
from loguru import logger
from sqlalchemy import select

from config import db_engine, db_settings
from db.db_model import RentApartments


def load_data_from_db() -> pd.DataFrame:
    """
    Load the rent_apartments table into a pd.DataFrame.

    Arguments:
        None

    Returns:
        pd.DataFrame: DataFrame of the dataset
    """
    logger.info(f'loading the table from {db_settings.db_conn_str}')
    query = select(RentApartments)
    return pd.read_sql(query, db_engine)

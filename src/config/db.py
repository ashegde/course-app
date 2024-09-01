"""
This module manages the database configuration.

The DBSettings class contains the settings and environment variables
assoociated with the database. This is done through Pydantic's BaseSettings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine


class DBSettings(BaseSettings):
    """
    A class that manages the database settings.

    This class manages the database settings, which are used when building
    the ML model. The relevant environment variables are loaded from an
    .env file.

    Attributes:
        db_conn_str: Connection to SQL database containing data.
        rent_apartments_table_name: SQL database table name
    """

    model_config = SettingsConfigDict(
        env_file='config/.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    db_conn_str: str
    rent_apartments_table_name: str


# initialize settings and logging
db_settings = DBSettings()

# creating the database engine
db_engine = create_engine(db_settings.db_conn_str)

import pandas as pd
from meteostat import Point, Daily
import logging

from src.common import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def fetch_daily_data(aqi_df: pd.DataFrame):
    # Coordinates of Poprad-Tatry Airport (LZTT) meteo station.
    location = Point(49.07, 20.24, 718)

    datetime_pd = aqi_df.index
    start_date = datetime_pd.min()
    end_date = datetime_pd.max()

    daily_weather = Daily(location, start_date, end_date)
    weather_df = daily_weather.fetch()

    # No data available for these features from selected meteo station
    weather_df.drop(["wdir", "tsun"], axis=1, inplace=True)

    return weather_df


def clean_missing_values(weather_df: pd.DataFrame):
    LOGGER.debug(f"Missing values:\n{weather_df.isna().sum()}")
    # N/A here means that there was no rain or snow - so filling with 0 instead
    weather_df["prcp"] = weather_df["prcp"].fillna(0.0)
    weather_df["snow"] = weather_df["snow"].fillna(0.0)

    # Let's assume that rest N/A is just missing reading, and fill with median
    weather_df.fillna(weather_df.median(), inplace=True)
    LOGGER.debug(f"Missing values after cleaning:\n{weather_df.isna().sum()}")

    return weather_df

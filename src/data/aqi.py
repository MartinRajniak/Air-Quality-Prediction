import pandas as pd
import hopsworks
import os
import logging
import datetime

from src.common import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)

def load_data(project_root):
    # TODO: stop using historical data once we have enough of hourly data
    hist_aqi_df = _load_historical_data(project_root)
    current_aqi_df = _load_current_data()
    aqi_df = pd.concat([hist_aqi_df, current_aqi_df], axis=0)
    return aqi_df

def _load_historical_data(project_root):
    aqi_history_path = os.path.join(project_root, "data", "air_quality_history.csv")
    aqi_df = pd.read_csv(aqi_history_path, skipinitialspace=True)
    aqi_df["date"] = pd.to_datetime(aqi_df["date"])
    aqi_df = aqi_df.set_index("date")
    aqi_df = aqi_df.sort_index()
    return aqi_df

def _load_current_data():
    hopsworks_aqi_token = os.environ["HOPSWORKS_AQI_TOKEN"]
    project = hopsworks.login(api_key_value=hopsworks_aqi_token)
    feature_store = project.get_feature_store()

    iaqi_fg = feature_store.get_feature_group(
        name="iaqi",
        version=1
    )

    iaqi_fg_df = iaqi_fg.select(["event_timestamp", "pm25", "pm10", "no2", "so2", "co"]).read()
    # Remove TimeZone info and reset to 0 hours so that it can be compared to historical data
    iaqi_fg_df["event_timestamp"] = pd.to_datetime(iaqi_fg_df["event_timestamp"]).dt.tz_localize(None).dt.normalize()
    # TODO: historical data is rounded but without rounding we might get better predictions (especially for CO)
    iaqi_fg_df = iaqi_fg_df.groupby(iaqi_fg_df["event_timestamp"], as_index=True).mean()
    iaqi_fg_df = iaqi_fg_df.sort_index()

    return iaqi_fg_df

def clean_missing_dates(aqi_df: pd.DataFrame):
    full_range = pd.date_range(start=aqi_df.index.min(), end=aqi_df.index.max(), freq="D")
    missing_dates = full_range.difference(aqi_df.index)
    LOGGER.debug(f"Missing dates: {missing_dates}")

    for missing_date in missing_dates:
        aqi_df.loc[missing_date] = aqi_df.loc[missing_date - datetime.timedelta(days=1)]
    aqi_df = aqi_df.sort_index()

    LOGGER.debug(f"Missing dates after cleanup: {full_range.difference(aqi_df.index)}")

    return aqi_df

def clean_missing_values(aqi_df: pd.DataFrame):
    LOGGER.debug(f"Missing values:\n{aqi_df.isna().sum()}")
    # For IAQI values it is OK to use average
    aqi_df = aqi_df.fillna(aqi_df.median())
    LOGGER.debug(f"Missing values after cleaning:\n{aqi_df.isna().sum()}")

    return aqi_df
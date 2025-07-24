import pandas as pd
import hopsworks
import os

def load_aqi_data():
    # TODO: stop using historical data once we have enough of hourly data
    hist_aqi_df = load_historical_aqi_data()
    current_aqi_df = load_current_aqi_data()
    return pd.concat([hist_aqi_df, current_aqi_df], axis=0)

def load_historical_aqi_data():
    aqi_df = pd.read_csv("../../data/air_quality_history.csv", skipinitialspace=True)
    aqi_df["date"] = pd.to_datetime(aqi_df["date"], format="%Y/%m/%d").dt.date
    aqi_df = aqi_df.set_index("date")
    aqi_df = aqi_df.sort_index()
    return aqi_df

def load_current_aqi_data():
    hopsworks_aqi_token = os.environ["HOPSWORKS_AQI_TOKEN"]
    project = hopsworks.login(api_key_value=hopsworks_aqi_token)
    feature_store = project.get_feature_store()

    iaqi_fg = feature_store.get_feature_group(
        name="iaqi",
        version=1
    )

    iaqi_fg_df = iaqi_fg.select(["event_timestamp", "pm25", "pm10", "no2", "so2", "co"]).read()
    iaqi_fg_df["event_timestamp"] = pd.to_datetime(iaqi_fg_df["event_timestamp"], format="%Y/%m/%d").dt.date
    # TODO: historical data is rounded but without rounding we might get better predictions (especially for CO)
    iaqi_fg_df = iaqi_fg_df.groupby(iaqi_fg_df["event_timestamp"], as_index=True).mean()
    iaqi_fg_df = iaqi_fg_df.sort_index()

    return iaqi_fg_df
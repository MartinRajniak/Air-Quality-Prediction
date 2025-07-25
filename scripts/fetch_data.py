import requests
import json
import os
import hopsworks
from hsfs.feature import Feature
from datetime import datetime
import pandas as pd
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.append(PROJECT_ROOT)

from src.common import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def fetch_current_iaqi():
    aqi_token = os.environ["AQI_TOKEN"]
    res = requests.get(
        f"https://api.waqi.info/feed/slovakia/poprad/zeleznicna/?token={aqi_token}"
    )
    response = json.loads(res.text)

    if response["status"] == "error":
        return f"Request error: {response["message"]}"

    date_time = datetime.fromisoformat(response["data"]["time"]["iso"])
    iaqi_data = {name: data["v"] for name, data in response["data"]["iaqi"].items()}

    current_iaqi_df = pd.DataFrame([iaqi_data], dtype="float32")
    current_iaqi_df.insert(0, "event_timestamp", date_time)

    return current_iaqi_df


def save_to_feature_store(current_iaqi_df):
    hopsworks_aqi_token = os.environ["HOPSWORKS_AQI_TOKEN"]
    project = hopsworks.login(api_key_value=hopsworks_aqi_token)
    feature_store = project.get_feature_store()

    features = [
        Feature(
            name="event_timestamp",
            type="timestamp",
            description="Timestamp of the event",
        ),
        Feature(
            name="co",
            type="float",
            description="Individual AQI for Carbon Monoxide (CO).",
        ),
        Feature(name="dew", type="float", description="Dew Point temperature."),
        Feature(name="h", type="float", description="Relative Humidity."),
        Feature(
            name="no2",
            type="float",
            description="Individual AQI for Nitrogen Dioxide (NO2).",
        ),
        Feature(name="p", type="float", description="Atmospheric Pressure."),
        Feature(
            name="pm10",
            type="float",
            description="Individual AQI for Particulate Matter (PM10).",
        ),
        Feature(
            name="pm25",
            type="float",
            description="Individual AQI for Particulate Matter (PM2.5).",
        ),
        Feature(
            name="so2",
            type="float",
            description="Individual AQI for Sulfur Dioxide (SO2).",
        ),
        Feature(name="t", type="float", description="Temperature."),
        Feature(name="w", type="float", description="Wind Speed."),
    ]

    iaqi_fg = feature_store.get_or_create_feature_group(
        name="iaqi",
        version=1,  # TODO: update when schema changes
        description="Individual AQI and weather hourly data",
        primary_key=["event_timestamp"],
        event_time="event_timestamp",
        online_enabled=False,  # Data used for training don't have to have low latency
        features=features,
    )

    LOGGER.info(
        f"Feature Group '{iaqi_fg.name}' (version {iaqi_fg.version}) retrieved or created successfully."
    )

    try:
        iaqi_fg.insert(current_iaqi_df)
        return "Successfully updated feature store"
    except Exception as e:
        LOGGER.error(f"Failed to update feature group: {e}")
        return "Failed to update feature group: insertion error"


if __name__ == "__main__":
    # TODO: comment out for production
    # LOGGER.setLevel(logging.DEBUG)

    LOGGER.info("Fetching current IAQI values...")
    result = fetch_current_iaqi()
    if not isinstance(result, pd.DataFrame):
        LOGGER.error(f"Failed to fetch current IAQI values: {result}")
        sys.exit(1)

    LOGGER.info("Successfully fetched current IAQI values.")
    LOGGER.debug(f"Current IAQI values:\n{result}")
    current_iaqi_df = result

    LOGGER.info("Saving current IAQI values to feature store...")
    result = save_to_feature_store(current_iaqi_df)
    LOGGER.info(result)

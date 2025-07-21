import requests
import json
import os
import hopsworks
from datetime import datetime


def fetch_current_aqi():
    aqi_token = os.environ["AQI_TOKEN"]
    res = requests.get(
        f"https://api.waqi.info/feed/slovakia/poprad/zeleznicna/?token={aqi_token}"
    )
    response = json.loads(res.text)

    if response["status"] == "error":
        return f"Request error: {response["message"]}"

    response_time_iso = response["data"]["time"]["iso"]
    date_time = datetime.fromisoformat(response_time_iso)
    hour = date_time.hour
    day = date_time.day
    month = date_time.month
    year = date_time.year

    aqi = response["data"]["aqi"]

    return hour, day, month, year, aqi

def save_to_feature_store():
    hopsworks_aqi_token = os.environ["HOPSWORKS_AQI_TOKEN"]

    project = hopsworks.login(api_key_value=hopsworks_aqi_token)
    feature_store = project.get_feature_store()

if __name__ == "__main__":
    result = fetch_current_aqi()
    print(result)

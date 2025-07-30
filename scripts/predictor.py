import joblib
import hopsworks
import numpy as np
import pandas as pd
import os
import sys

print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
print(f"Available packages:")
import pkg_resources

installed_packages = [d.project_name for d in pkg_resources.working_set]
print(installed_packages)

# Add sources to the path
MODEL_FILES_PATH = os.environ["MODEL_FILES_PATH"]
sys.path.append(MODEL_FILES_PATH)

from src.data import aqi, meteo
from src.data.calendar import add_calendar_features
from src.data.features import _flatten_windows


class Predictor:
    def __init__(self):
        model_dir = os.environ["MODEL_FILES_PATH"]
        self.model = joblib.load(os.path.join(model_dir, "aqi_prediction_model.pkl"))
        self.feature_scaler = joblib.load(os.path.join(model_dir, "feature_scaler.bin"))

    def predict(self, _):
        # Once model is trained, these are fixed and we have to remember them
        # TODO: remember and reuse
        historical_window_size = 3
        prediction_window_size = 3
        num_of_predictions = 1

        # Prepare historical data
        # TODO: this is same as in train_model script - reuse
        # TODO: store processed features in feature store (together with historical)

        # aqi_df = aqi._load_current_data()
        project = hopsworks.login()
        feature_store = project.get_feature_store()
        iaqi_fg = feature_store.get_feature_group(name="iaqi", version=1)

        iaqi_fg_df = iaqi_fg.select(
            ["event_timestamp", "pm25", "pm10", "no2", "so2", "co"]
        ).read()
        # Remove TimeZone info and reset to 0 hours so that it can be compared to historical data
        iaqi_fg_df["event_timestamp"] = (
            pd.to_datetime(iaqi_fg_df["event_timestamp"])
            .dt.tz_localize(None)
            .dt.normalize()
        )

        iaqi_fg_df = iaqi_fg_df.groupby(
            iaqi_fg_df["event_timestamp"], as_index=True
        ).mean()
        iaqi_fg_df = iaqi_fg_df.sort_index()
        aqi_df = iaqi_fg_df

        aqi_df = aqi.clean_missing_dates(aqi_df)
        aqi_df = aqi.clean_missing_values(aqi_df)
        aqi_df = add_calendar_features(aqi_df)
        aqi_df.index = aqi_df.index.astype("datetime64[ns]")

        weather_df = meteo.fetch_daily_data(aqi_df)
        weather_df = meteo.clean_missing_values(weather_df)
        weather_df.index = weather_df.index.astype("datetime64[ns]")

        merged_df = pd.merge_asof(aqi_df, weather_df, left_index=True, right_index=True)
        merged_df = merged_df.astype(float)

        X = merged_df[-historical_window_size:]
        X = self.feature_scaler.transform(X)

        for day in range(num_of_predictions):
            # Input expects multiple windows
            X_flat = _flatten_windows([X])

            y_pred = self.model.predict(X_flat)

            # Split y_pred into 3 arrays, one for each prediction day
            y_pred_split = np.split(y_pred.flatten(), prediction_window_size)

            # Create DataFrame for predictions, each row is a prediction day
            predictions_df = pd.DataFrame(y_pred_split, columns=merged_df.columns)

            # Set the index to continue from the last date in merged_df
            last_date = X.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=prediction_window_size,
                freq="D",
            )
            predictions_df.index = future_dates

            # Add predictions to the end so that we can use them as input (don't forget to remove the same number of items as we added - model expects certain size)
            X = pd.concat([X[prediction_window_size:], predictions_df], axis=0)

        X = self.feature_scaler.inverse_transform(X)

        # Definition of Air Quality Index is maximum value of Individual Air Quality Indexes
        iaqi_features = ["pm25", "pm10", "no2", "so2", "co"]
        X["aqi"] = X[iaqi_features].max(axis=1)

        result = X[-num_of_predictions * prediction_window_size :][iaqi_features]
        return result.to_json()

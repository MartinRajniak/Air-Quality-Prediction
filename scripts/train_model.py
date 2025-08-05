import logging
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.append(PROJECT_ROOT)

from src.data import aqi, meteo
from src.data.calendar import add_calendar_features
from src.data.features import FeatureScaler, split_to_windows, flatten_windows
from src.model.training import split_data
from src.model.evaluation import evaluate_iaqi_predictions, create_metrics_dataframe, get_day_n_metrics
from src.model.inference import recursive_forecasting
from src.hopsworks.client import HopsworksClient
from src.model import xgboost
from src.common import LOGGER_NAME

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

LOGGER = logging.getLogger(LOGGER_NAME)

if __name__ == "__main__":
    # How many (lagged) days to use as input during training
    historical_window_size = 3
    # How many days to teach the model to predict
    prediction_window_size = 3
    # How many predictions to do as part of recursive forecasting
    num_of_predictions = 1

    # TODO: comment out for production
    LOGGER.setLevel(logging.DEBUG)

    LOGGER.info("Loading AQI data...")
    aqi_df = aqi.load_data(PROJECT_ROOT)
    LOGGER.debug(aqi_df.head())

    LOGGER.info("Cleaning missing dates...")
    aqi_df = aqi.clean_missing_dates(aqi_df)

    LOGGER.info("Cleaning missing values...")
    aqi_df = aqi.clean_missing_values(aqi_df)

    LOGGER.info("Adding calendar features...")
    aqi_df = add_calendar_features(aqi_df)
    LOGGER.debug(aqi_df.head())

    # TODO: Fetch daily for now (because of historical AQI data)
    # but once we have hourly AQI data, download hourly meteo as well
    LOGGER.info("Fetching meteo features...")
    weather_df = meteo.fetch_daily_data(aqi_df)
    weather_df = meteo.clean_missing_values(weather_df)
    LOGGER.debug(weather_df.head())

    LOGGER.info("Merging AQI and METEO data...")
    merged_df = pd.merge_asof(aqi_df, weather_df, left_index=True, right_index=True)
    # Keep same format - ML models work best with Float values
    merged_df = merged_df.astype(float)
    LOGGER.debug(merged_df.head())

    target_columns = merged_df.columns

    train_df, val_df, test_df = split_data(merged_df)

    LOGGER.info("Scaling features...")
    feature_scaler = FeatureScaler()
    feature_scaler.fit(train_df)

    train_df = feature_scaler.transform(train_df)
    val_df = feature_scaler.transform(val_df)
    test_df = feature_scaler.transform(test_df)
    LOGGER.debug(train_df.head())

    LOGGER.info("Creating lagged feature windows...")
    (
        X_window_train,
        X_window_val,
        X_window_test,
        y_window_train,
        y_window_val,
        y_window_test,
    ) = split_to_windows(
        train_df, val_df, test_df, historical_window_size, prediction_window_size, target_columns=target_columns
    )
    LOGGER.debug(f"Last X window:\n{X_window_test[-1]}")
    LOGGER.debug(f"Last y window:\n{y_window_test[-1]}")

    LOGGER.info("Creating flat window features for model...")
    X_flat_train, X_flat_val, X_flat_test, y_flat_train, y_flat_val, y_flat_test = (
        flatten_windows(
            X_window_train,
            X_window_val,
            X_window_test,
            y_window_train,
            y_window_val,
            y_window_test,
        )
    )
    LOGGER.debug(f"Last X sample:\n{X_flat_test[-1]}")
    LOGGER.debug(f"Last y sample:\n{y_flat_test[-1]}")

    LOGGER.info(f"Fitting model...")
    model = xgboost.create_regressor()
    model.fit(X_flat_train, y_flat_train)

    LOGGER.info(f"Evaluating model...")
    actual, predictions = recursive_forecasting(model, test_df, historical_window_size, prediction_window_size, num_of_predictions)

    predictions = [feature_scaler.inverse_transform(prediction) for prediction in predictions]
    actual = [feature_scaler.inverse_transform(value) for value in actual]

    prediction_metrics = evaluate_iaqi_predictions(y_true=actual, y_pred=predictions, prediction_window_size=prediction_window_size, num_of_predictions=num_of_predictions)
    prediction_metrics_df = create_metrics_dataframe(prediction_metrics)
    LOGGER.debug(prediction_metrics_df.to_string(float_format="%.2f"))

    LOGGER.info(f"Saving model to model registry...")
    # Using recursive forecasting - error is cumulative - cannot improve Day 3 prediction without improving Day 1 - so last day's results are enough
    last_day_metrics = get_day_n_metrics(prediction_metrics, prediction_window_size * num_of_predictions)
    # Using single metric for model comparison - The Willmott index - it gives credit for correlation but heavily penalizes systematic errors that would make the forecasts unreliable for air quality management.
    metrics = last_day_metrics["Willmott"]
    hopsworks_model = HopsworksClient().save_model(PROJECT_ROOT, model, metrics, X_flat_test[0], y_flat_test[0], feature_scaler)
    LOGGER.debug(f"Hopsworks Model:\n{hopsworks_model.description}")

    LOGGER.info(f"Deploying model...")
    deployment = HopsworksClient().deploy_model(hopsworks_model, overwrite=False)
    LOGGER.debug(f"Hopsworks Deployment:\n{deployment}")

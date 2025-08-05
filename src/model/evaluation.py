import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import logging

from src.common import LOGGER_NAME, IAQI_FEATURES

LOGGER = logging.getLogger(LOGGER_NAME)


# Index of Agreement (Willmott) - combines correlation and bias
def index_of_agreement(y_true, y_pred):
    mean_obs = np.mean(y_true)
    numerator = np.sum((y_pred - y_true) ** 2)
    denominator = np.sum((np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs)) ** 2)
    return 1 - (numerator / denominator)


def pearson_value(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def spearman_value(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


# Mean Bias Error
def mean_bias_error(y_true, y_pred):
    return np.mean(np.array(y_true) - np.array(y_pred))


# Mean Absolute Bias Error
def mean_abs_bias_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


# Normalized Mean Bias Error
def norm_mean_bias_error(y_true, y_pred):
    return mean_bias_error(y_true, y_pred) / np.mean(np.array(y_true)) * 100


# Mean Absolute Percentage Error (be careful with near-zero values)
def mape(y_true, y_pred, epsilon=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


# Symmetric Mean Absolute Percentage Error (better for values near zero)
def smape(y_true, y_pred):
    return 100 * np.mean(
        np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    )


# RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Normalized RMSE
def nrmse(y_true, y_pred):
    return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true))


# Fractional Bias (common in air quality modeling)
def fractional_bias(y_true, y_pred):
    return 2 * np.mean(y_pred - y_true) / (np.mean(y_pred) + np.mean(y_true))


# Fractional Gross Error
def fractional_gross_error(y_true, y_pred):
    return 2 * np.mean(np.abs(y_pred - y_true)) / (np.mean(y_pred) + np.mean(y_true))


# Factor of 2 (FAC2) - fraction of predictions within factor of 2
def factor_of_2(y_true, y_pred, epsilon=1e-8):
    ratio = y_pred / (y_true + epsilon)
    return np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100


METRIC_FUNCTIONS = {
    "RMSE": rmse,
    "NRMSE": nrmse,
    "MAE": mean_absolute_error,
    "R2": r2_score,
    "Pearson": pearson_value,
    "Spearman": spearman_value,
    "Willmott": index_of_agreement,
    "Mean Bias Error": mean_bias_error,
    "Mean Absolute Bias Error": mean_abs_bias_error,
    "Normalized Mean Bias Error": norm_mean_bias_error,
    "Fractional Bias": fractional_bias,
    "Fractional Gross Error": fractional_gross_error,
    "Factor of 2": factor_of_2,
}


def evaluate_iaqi_predictions(y_true, y_pred, prediction_window_size, num_of_predictions):
    split_data = _split_by_iaqi(
        y_true, y_pred, prediction_window_size, num_of_predictions
    )

    return {
        metric_name: {
            iaqi: [
                # 4 decimal points is enough for comparison
                round(metric_function(y_true_day, y_pred_day), 4)
                for y_true_day, y_pred_day in zip(y_true_by_day, y_pred_by_day)
            ]
            for iaqi, (y_true_by_day, y_pred_by_day) in split_data.items()
        }
        for metric_name, metric_function in METRIC_FUNCTIONS.items()
    }


def _split_by_iaqi(y_true, y_pred, prediction_window_size, num_of_predictions):
    return {
        iaqi_feature: (
            [
                np.array([true_value[iaqi_feature].iloc[day] for true_value in y_true])
                for day in range(prediction_window_size * num_of_predictions)
            ],
            [
                np.array([prediction[iaqi_feature].iloc[day] for prediction in y_pred])
                for day in range(prediction_window_size * num_of_predictions)
            ],
        )
        for iaqi_feature in IAQI_FEATURES
    }


def create_metrics_dataframe(prediction_metrics):
    # This will hold the flattened data
    data = []
    
    # Iterate through the nested dictionary
    for metric_name, iaqi_data in prediction_metrics.items():
        for iaqi, metric_values in iaqi_data.items():
            # Create a row for each IAQI feature
            row = {'Metric': metric_name, 'IAQI_Feature': iaqi}
            # Add a column for each day
            for day, value in enumerate(metric_values, 1):
                row[f'Day_{day}'] = value
            data.append(row)
            
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    # Set a multi-level index for better grouping
    df = df.set_index(['Metric', 'IAQI_Feature'])
    
    return df

def get_day_n_metrics(all_day_metrics_result, day_number):
    day_index = day_number - 1
    
    return {
        metric_name: {
            iaqi: values[day_index] if len(values) > day_index else None
            for iaqi, values in iaqi_data.items()
        }
        for metric_name, iaqi_data in all_day_metrics_result.items()
    }
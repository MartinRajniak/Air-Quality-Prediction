import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import logging

from src.common import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def evaluate_predictions(y_true, y_pred, window_size):
    n_outputs = y_true.shape[1]
    n_outputs_per_day = n_outputs // window_size

    result = []
    for window_index in range(window_size):
        window_result = []
        for output_index in range(n_outputs_per_day):
            i = window_index * n_outputs_per_day + output_index

            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            r2 = r2_score(y_true[:, i], y_pred[:, i])

            window_result.append({"RMSE": rmse, "R2_Score": r2})
        result.append(window_result)
    return result


def print_prediction_metrics(prediction_metrics, target_columns):
    for day_index, output_metrics in enumerate(prediction_metrics):
        LOGGER.info(f"Day {day_index+1}:")
        for output_index, metrics in enumerate(output_metrics):
            LOGGER.info(f"\tMetrics for {target_columns[output_index]}:")
            for metric, value in metrics.items():
                LOGGER.info(f"\t {metric}: {value:.4f}")
            LOGGER.info("-" * 20)

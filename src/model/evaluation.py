import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

from src.common import LOGGER_NAME

LOGGER = logging.getLogger(LOGGER_NAME)


def evaluate_predictions(y_true, y_pred, window_size, original_columns):
    n_outputs = y_true.shape[1]

    for window_index in range(window_size):
        LOGGER.info(f"Day {window_index+1}:")
        for output_index in range(n_outputs // window_size):
            i = output_index + window_size * window_index

            LOGGER.info(
                f"Metrics for Output {output_index+1} ({original_columns[output_index]}):"
            )
            LOGGER.info(f"  MAE: {mean_absolute_error(y_true[:, i], y_pred[:, i]):.4f}")
            LOGGER.info(f"  MSE: {mean_squared_error(y_true[:, i], y_pred[:, i]):.4f}")
            LOGGER.info(
                f"  RMSE: {np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])):.4f}"
            )
            LOGGER.info(f"  R2 Score: {r2_score(y_true[:, i], y_pred[:, i]):.4f}")
            LOGGER.info("-" * 20)

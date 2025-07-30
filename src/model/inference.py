import pandas as pd
from pandas import DataFrame
import numpy as np

import torch

from src.data.features import _flatten_windows, _split_to_windows

# TODO: support case of forecasting into the future (for real world predictions)
def recursive_forecasting(model, input: DataFrame, historical_window_size, prediction_window_size, num_of_predictions, torch=False):
    true_values = []
    predictions = []
    # With recursive forecasting input and target need to have same columns
    target_columns = input.columns
    input_windows, _ = _split_to_windows(input, historical_window_size, prediction_window_size, target_columns)
    for input_window in input_windows:
        if (input_window.index.max() >= (input.index.max() - pd.Timedelta(days=prediction_window_size*num_of_predictions-1))):
            # Prediction window doesn't take into account number of predictions - so we need to stop early
            break

        target_window = input.iloc[:0].copy()
        for day_index in range(num_of_predictions):
            # Model expects multiple windows
            if torch:
                y_pred = torch_predict(model, [input_window])
            else:
                y_pred = predict(model, [input_window])

            # Split y_pred into 3 arrays, one for each prediction day
            y_pred_split = np.split(y_pred.flatten(), prediction_window_size)

            # Create DataFrame for predictions, each row is a prediction day
            predictions_df = pd.DataFrame(y_pred_split, columns=target_columns)

            # Set the index to continue from the last date in merged_df
            last_date = input_window.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_window_size, freq="D")
            predictions_df.index = future_dates

            # Add predictions to the end so that we can use them as input (don't forget to remove the same number of items as we added - model expects certain size)
            input_window = pd.concat([input_window[prediction_window_size:], predictions_df], axis=0)

            target_window = pd.concat([target_window, input.loc[future_dates]], axis=0)

        predictions.append(input_window)
        true_values.append(target_window)

    return true_values, predictions

def predict(model, input_windows):
    X_flat = _flatten_windows(input_windows)
    y_pred = model.predict(X_flat)
    return y_pred

def torch_predict(model, input_windows):
    X_lstm_test = windows_to_tensor(input_windows)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_lstm_test).cpu().numpy()
    
    return y_pred

def windows_to_tensor(windows):
    tensor = torch.tensor(
        np.stack([window.values for window in windows]), dtype=torch.float32
    )
    return tensor
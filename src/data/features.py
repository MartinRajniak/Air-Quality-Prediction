import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureScaler:

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, train_df, val_df, test_df):
        self.numerical_features = [
            col
            for col in train_df.columns
            if col not in ["is_leap_year", "is_feb29", "is_working_day"]
        ]

        # TODO: use one-hot encoding on features like year, day_of_week, etc.

        # Important: Fit the scaler only on the training data 
        # and then transform both training and testing data to prevent data leakage.
        self.scaler = self.scaler.fit(
            pd.concat(
                [train_df[self.numerical_features], val_df[self.numerical_features]],
                axis=0,
            )
        )

        train_df.loc[:, self.numerical_features] = self.scaler.transform(
            train_df[self.numerical_features]
        )
        val_df.loc[:, self.numerical_features] = self.scaler.transform(
            val_df[self.numerical_features]
        )
        test_df.loc[:, self.numerical_features] = self.scaler.transform(
            test_df[self.numerical_features]
        )

        return train_df, val_df, test_df
    
    def transform(self, dataframe):
        dataframe[self.numerical_features] = self.scaler.transform(dataframe[self.numerical_features])
        return dataframe
    
    def inverse_transform(self, dataframe):
        dataframe[self.numerical_features] = self.scaler.inverse_transform(dataframe[self.numerical_features])
        return dataframe


def split_to_windows(train_df, val_df, test_df, historical_window_size, prediction_window_size):
    X_window_train, y_window_train = _split_to_windows(train_df, historical_window_size, prediction_window_size)
    X_window_val, y_window_val = _split_to_windows(val_df, historical_window_size, prediction_window_size)
    X_window_test, y_window_test = _split_to_windows(test_df, historical_window_size, prediction_window_size)

    return X_window_train, X_window_val, X_window_test, y_window_train, y_window_val, y_window_test

def _split_to_windows(X, historical_window_size, prediction_window_size):
    original_input_size = X.shape[0]

    X_new = []
    y_new = []
    for i in range(original_input_size - historical_window_size - prediction_window_size + 1):
        X_window = X[i : i + historical_window_size]
        X_new.append(X_window)

        y_window = X[i + historical_window_size : i + historical_window_size + prediction_window_size]
        y_new.append(y_window)
    
    return X_new, y_new

def flatten_windows(X_window_train, X_window_val, X_window_test, y_window_train, y_window_val, y_window_test):
    X_flat_train, y_flat_train = _flatten_windows(X_window_train), _flatten_windows(y_window_train)
    X_flat_val, y_flat_val = _flatten_windows(X_window_val), _flatten_windows(y_window_val)
    X_flat_test, y_flat_test = _flatten_windows(X_window_test), _flatten_windows(y_window_test)
    
    return X_flat_train, X_flat_val, X_flat_test, y_flat_train, y_flat_val, y_flat_test

def _flatten_windows(windows):
    # Regressors require 2D features - more columns instead of more dimensions
    return np.array([window.values.flatten() for window in windows])
import time
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

def grid_search(train_var_pred, train_targets, val_var_pred, val_targets, param_grid, stop_training_flag=None):
    """
    Performs a grid search to optimize hyperparameters for a variational recurrent neural network (VarNN).

    Parameters:
        train_var_pred (np.ndarray): Training input features.
        train_targets (np.ndarray): Training target values.
        val_var_pred (np.ndarray): Validation input features.
        val_targets (np.ndarray): Validation target values.
        param_grid (dict): Dictionary containing hyperparameter lists:
            - 'learning_rate' (list of float): Learning rates to try.
            - 'batch_size' (list of int): Batch sizes to try.
            - 'hidden_layer_sizes' (list of tuples): Different hidden layer configurations.
            - 'epoch' (list of int): Number of epochs to train for.
        stop_training_flag (threading.Event, optional): Event flag to interrupt the grid search.

    Returns:
        dict: Best hyperparameters found.
        float: Corresponding lowest mean squared error.
        float: Execution time in seconds.
    """
    if train_var_pred.shape[0] != train_targets.shape[0]:
        raise ValueError("Mismatch between training features and targets dimensions.")
    if val_var_pred.shape[0] != val_targets.shape[0]:
        raise ValueError("Mismatch between validation features and targets dimensions.")
    
    required_keys = {'learning_rate', 'batch_size', 'hidden_layer_sizes', 'epoch'}
    if not required_keys.issubset(param_grid.keys()):
        raise ValueError(f"Parameter grid must contain {required_keys}.")

    best_params = None
    best_mse = float("inf")
    start_time = time.time()
    
    for lr in param_grid['learning_rate']:
        if stop_training_flag is not None and stop_training_flag.is_set():
            st.warning("Grid search interrupted (stop flag set).")
            return best_params, best_mse, time.time() - start_time
        for batch_size in param_grid['batch_size']:
            if stop_training_flag is not None and stop_training_flag.is_set():
                st.warning("Grid search interrupted (stop flag set).")
                return best_params, best_mse, time.time() - start_time
            for hidden_layer_sizes in param_grid['hidden_layer_sizes']:
                if stop_training_flag is not None and stop_training_flag.is_set():
                    st.warning("Grid search interrupted (stop flag set).")
                    return best_params, best_mse, time.time() - start_time
                for epoch in param_grid['epoch']:
                    if stop_training_flag is not None and stop_training_flag.is_set():
                        st.warning("Grid search interrupted (stop flag set).")
                        return best_params, best_mse, time.time() - start_time

                    model = build_varnn(
                        input_dim=train_var_pred.shape[1],
                        output_dim=train_targets.shape[1],
                        learning_rate=lr,
                        hidden_layer_sizes=hidden_layer_sizes
                    )
                    model.fit(
                        train_var_pred,
                        train_targets,
                        epochs=epoch,
                        batch_size=batch_size,
                        validation_data=(val_var_pred, val_targets),
                        verbose=0
                    )
                    
                    val_pred = model.predict(val_var_pred)
                    mse = mean_squared_error(val_targets, val_pred)
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'hidden_layer_sizes': hidden_layer_sizes,
                            'epoch': epoch
                        }
    
    execution_time = time.time() - start_time
    return best_params, best_mse, execution_time


def build_varnn(input_dim: int, output_dim: int, learning_rate: float, hidden_layer_sizes):
    """
    Builds and compiles a Variational Recurrent Neural Network (VarNN) model.
    
    Parameters:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output.
        learning_rate (float): Learning rate for the optimizer.
        hidden_layer_sizes (int or list of int): Number of neurons in hidden layers.
    
    Returns:
        Sequential: Compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    if isinstance(hidden_layer_sizes, int):
        hidden_layer_sizes = [hidden_layer_sizes]
    
    for size in hidden_layer_sizes:
        model.add(Dense(units=size, activation='relu'))
        model.add(Dropout(rate=0.2))
    
    model.add(Dense(units=output_dim))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_var_predictions(data, model, lag_order: int, features: list):
    """
    Generates predictions using a Vector AutoRegression (VAR) model.
    
    Parameters:
        data (pd.DataFrame): Input dataset containing time-series features.
        model: Pre-trained VAR model for forecasting.
        lag_order (int): Number of lagged observations to consider.
        features (list): List of feature column names used for prediction.
    
    Returns:
        np.ndarray: Array of predicted values.
    """
    lagged_data = []
    
    for i in range(lag_order, len(data)):
        pred = model.forecast(data[features].values[i - lag_order:i], steps=1)
        lagged_data.append(pred[0])
    
    return np.array(lagged_data)


def create_lagged_features(df, lags):
    """Generates lagged features for time series data.

    Args:
        df (pd.DataFrame): Input time series dataset with a datetime index.
        lags (int): Number of lagged periods to generate.

    Returns:
        pd.DataFrame: Dataset with original and lagged features, with NaN values dropped.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    df_lagged = df.copy()

    for lag in range(1, lags + 1):
        lagged = df.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
        df_lagged = pd.concat([df_lagged, lagged], axis=1)

    return df_lagged.dropna()

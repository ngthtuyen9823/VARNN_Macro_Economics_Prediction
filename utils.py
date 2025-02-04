# Standard libraries
import time

# Machine learning and preprocessing
from sklearn.metrics import mean_squared_error

# Data manipulation and visualization
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf

from models import build_varnn

def grid_search(train_var_pred, train_targets, val_var_pred, val_targets, param_grid):
    best_params = None
    best_mse = float("inf")
    start_time = time.time()

    if train_var_pred.shape[0] != train_targets.shape[0]:
        raise ValueError("Mismatch between training features and targets dimensions.")
    if val_var_pred.shape[0] != val_targets.shape[0]:
        raise ValueError("Mismatch between validation features and targets dimensions.")
    
    if not all(key in param_grid for key in ['learning_rate', 'batch_size', 'hidden_layer_sizes', 'epoch']):
        raise ValueError("Parameter grid must contain 'learning_rate', 'batch_size', 'hidden_layer_sizes', and 'epoch'.")

    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            for hidden_layer_sizes in param_grid['hidden_layer_sizes']:
                for epoch in param_grid['epoch']:
                    model = build_varnn(input_dim=train_var_pred.shape[1], 
                                        output_dim=train_targets.shape[1], 
                                        learning_rate=lr, 
                                        hidden_layer_sizes=hidden_layer_sizes)
                    model.fit(
                        train_var_pred, train_targets,
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


def add_gaussian_noise(time_series, mean, stddev):
    noise = np.random.normal(mean, stddev, len(time_series))
    noisy_series = time_series + noise
    return noisy_series

def generate_new_dates(df, periods):
        first_date = df.index.min()
        return pd.date_range(end=first_date - pd.DateOffset(years=1),
                             periods=periods,
                             freq='YS')
        
def augment_with_gaussian(data, mean=0.0, stddev=0.05):
    augmented_datasets = []
    augmented_data = data.copy()
        
    for column in data.columns:
        if np.issubdtype(data[column].dtype, np.number):
            augmented_data[column] = add_gaussian_noise(
                data[column].dropna(), mean, stddev
            )
        
    new_dates = generate_new_dates(data, len(data))
    augmented_data.index = new_dates
    augmented_datasets.append(augmented_data)
        
    return pd.concat([data] + augmented_datasets, axis=0).sort_index()


def augment_timeseries_data(data, n_periods=24):    
    new_dates = generate_new_dates(data, n_periods)
    augmented_data = []
    
    for column in data.columns:
        mean = data[column].mean()
        std = data[column].std()
        trend = np.polyfit(range(len(data)), data[column].values, 1)[0]
        
        base_trend = -np.arange(n_periods)[::-1] * trend + mean
        seasonality = np.sin(np.linspace(0, 2*np.pi, 12)) * std * 0.5
        noise = np.random.normal(0, std * 0.1, n_periods)
        
        new_series = pd.Series(
            base_trend + np.tile(seasonality, n_periods//12 + 1)[:n_periods] + noise,
            index=new_dates
        )
        augmented_data.append(new_series)
    
    return (pd.concat([pd.DataFrame(dict(zip(data.columns, augmented_data))), data])).sort_index()
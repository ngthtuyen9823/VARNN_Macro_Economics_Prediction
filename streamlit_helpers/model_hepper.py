import os
import json
import pickle

import pandas as pd
import numpy as np

import streamlit as st

from statsmodels.tsa.api import VAR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf


class StopTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, stop_training_flag):
        super().__init__()
        self.stop_training_flag = stop_training_flag

    def on_epoch_end(self, epoch, logs=None):
        if self.stop_training_flag.is_set():
            self.model.stop_training = True
            st.warning(f"Training stopped at epoch {epoch + 1}")


def save_grid_search_results(file_name, model_type, best_params, best_mse=None, search_time=None, best_lags=None):
    """Save grid search results in a JSON file."""
    grid_search_results = {
        'model_type': model_type,
        'best_parameters': best_params,
    }
    if best_mse is not None:
        grid_search_results['best_mse'] = best_mse
    if search_time is not None:
        grid_search_results['search_time'] = search_time
    if best_lags is not None:
        grid_search_results['best_lags'] = best_lags

    os.makedirs(f'results/{file_name}', exist_ok=True)
    with open(f'results/{file_name}/{model_type}_grid_search_results.json', 'w') as f:
        json.dump(grid_search_results, f)


def plot_training_history(history):
    """Plot training and validation loss along with final losses."""
    loss_df = pd.DataFrame({
        'Epoch': range(1, len(history['loss']) + 1),
        'Training Loss': history['loss'],
        'Validation Loss': history['val_loss']
    })
    st.line_chart(loss_df.set_index('Epoch'))
    st.write("### Final Losses")
    st.write(f"Final Training Loss: {history['loss'][-1]:.4f}")
    st.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")


def evaluate_predictions(ground_truth, predictions, execution_time):
    """Calculate evaluation metrics and return as a DataFrame."""
    test_mse = mean_squared_error(ground_truth, predictions)
    test_mae = mean_absolute_error(ground_truth, predictions)
    test_rmse = np.sqrt(test_mse)
    y_mean = np.mean(ground_truth)
    cv_rmse = test_rmse / y_mean
    evaluation_metrics = {
         "Metric": ["Test Time (seconds)", "Test MSE", "Test MAE", "Test RMSE", "Test CV RMSE"],
         "Value": [f"{execution_time:.2f}", f"{test_mse:.4f}", f"{test_mae:.4f}", f"{test_rmse:.4f}", f"{cv_rmse:.4f}"]
    }
    return pd.DataFrame(evaluation_metrics)


def display_last_saved_results(file_name, model_type):
    """Display the last saved grid search results and training history."""
    with open(f'results/{file_name}/{model_type}_grid_search_results.json', 'r') as f:
        grid_search_results = json.load(f)
    st.write("### Last Saved Results")
    st.json(grid_search_results.get('best_parameters', {}))
    st.write(f"Best MSE: {grid_search_results.get('best_mse', 'N/A')}")
    st.write(f"Search Time: {grid_search_results.get('search_time', 'N/A')}")
    st.write(f"Best Lags (VAR): {grid_search_results.get('best_lags', 'N/A')}")
    st.write("### Training History")
    with open(f'results/{file_name}/training_history.pkl', "rb") as f:
        loaded_history = pickle.load(f)
    plot_training_history(loaded_history)


def train_and_save_model(model, X_train, y_train, X_val, y_val, file_name, model_type, best_params, stop_training_flag):
    """
    Train the model, save training history and the final model.
    
    Args:
        model: The Keras model to train.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        file_name: Identifier for saving files.
        model_type: The model type (e.g., 'VARNN', 'Hybrid VARNN', 'VAR').
        best_params: Dictionary containing hyperparameters (including 'epoch' and 'batch_size').
        stop_training_flag: A threading.Event flag to stop training early.
    
    Returns:
        The training history object or None if training failed.
    """
    try:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=best_params['epoch'],
            batch_size=best_params['batch_size'],
            verbose=1,
            callbacks=[
                StopTrainingCallback(stop_training_flag),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            ]
        )
        os.makedirs(f'results/{file_name}', exist_ok=True)
        with open(f'results/{file_name}/training_history.pkl', "wb") as f:
            pickle.dump(history.history, f)
        if not stop_training_flag.is_set():
            os.makedirs(f'models/{file_name}', exist_ok=True)
            model.save(f'models/{file_name}/{model_type}_final_model.keras')
            st.success(f"{model_type} training completed and model saved.")
        return history
    except Exception as e:
        st.error(f"Training stopped or failed: {e}")
        return None

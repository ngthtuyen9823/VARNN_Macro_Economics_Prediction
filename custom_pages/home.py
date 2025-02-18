# Standard libraries
import os
import json
import threading
import time
import pickle

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Streamlit for UI and AgGrid for bảng dữ liệu có thể chỉnh sửa
import streamlit as st

# Statistical and time series analysis
from statsmodels.tsa.api import VAR

# Deep learning with TensorFlow
import tensorflow as tf

from helpers.models import (
    create_var_predictions,
    create_lagged_features,
    grid_search,
    build_varnn
)
from helpers.visualization import visualize_predictions, visualize_column
from streamlit_helpers.hepper import (
    upload_data,
    apply_preprocessing,
    create_directories,
    augment_data,
    stationarity_checks,
    normalize_data,
    split_data,
)
from streamlit_helpers.model_hepper import (
    save_grid_search_results,
    plot_training_history,
    evaluate_predictions,
    display_last_saved_results,
    train_and_save_model
)


def show_home_page():
    st.title("Macroeconomic Multivariate Forecasting Tool")

    # 1. Data Upload and Preprocessing
    data, file_name = upload_data()
    if data is None:
        st.info("Please upload a CSV file.")
        return

    directory_results, directory_models = create_directories(file_name)
    data = apply_preprocessing(data)
    st.subheader("Raw Data")
    st.dataframe(data)

    selected_col = st.sidebar.selectbox("Select Column to Visualize", data.columns)
    visualize_column(data, selected_col)

    # 2. Data Augmentation and Stationarity Check
    data = augment_data(data, selected_col)
    data = stationarity_checks(data, selected_col)

    # 3. Data Normalization
    data = normalize_data(data, selected_col)

    # 4. Data Splitting
    train_data, val_data, test_data = split_data(data)

    # 5. Model Training / Testing UI
    model_type = st.sidebar.selectbox("Select Model", ["VARNN", "Hybrid VARNN", "VAR"])
    train_button = st.sidebar.button("Train Model")
    stop_button = st.sidebar.button("Stop Training")
    test_button = st.sidebar.button("Test Model")
    # (Bỏ predict_button khỏi trang Home để chuyển sang trang Predict riêng biệt)

    stop_training = threading.Event()
    if stop_button:
        stop_training.set()
        st.warning("Stopping training process...")

    global flag
    flag = False

    if train_button:
        st.subheader("Parameter Optimization and Training")
        stop_training.clear()

        if model_type == "VARNN":
            st.subheader("Training VARNN Model")
            var_temp = VAR(train_data)
            var_temp_result = var_temp.fit(maxlags=5, ic="aic")
            best_lags = var_temp_result.k_ar
            st.write(f"Optimal lag order determined by VAR: {best_lags}")

            # Tạo tập dữ liệu với các giá trị lag
            train_lagged = create_lagged_features(train_data, best_lags)
            val_lagged = create_lagged_features(val_data, best_lags)
            test_lagged = create_lagged_features(test_data, best_lags)

            n_features = len(train_data.columns)
            X_train_vn = train_lagged.iloc[:, n_features:]
            y_train_vn = train_lagged.iloc[:, :n_features]
            X_val_vn = val_lagged.iloc[:, n_features:]
            y_val_vn = val_lagged.iloc[:, :n_features]

            # Grid search cho tìm hyperparameters tối ưu
            param_grid = {
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [16, 32, 64],
                "hidden_layer_sizes": [32, 64, 128],
                "epoch": [50, 100, 150],
            }

            best_params, best_mse, search_time = grid_search(
                X_train_vn,
                y_train_vn.values,
                X_val_vn,
                y_val_vn.values,
                param_grid,
                stop_training_flag=stop_training
            )

            st.success("Parameter optimization completed for VARNN.")
            st.write("**Best Parameters:**")
            st.json(best_params)
            st.write(f"**Best MSE:** {best_mse:.4f}")
            st.write(f"**Search Time:** {search_time:.2f} seconds")
            st.write(f"**Best Lags (VAR):** {best_lags}")

            # Lưu kết quả grid search và mô hình VAR
            save_grid_search_results(file_name, model_type, best_params, best_mse, search_time, best_lags)
            with open(f"results/{file_name}/var_result.pkl", "wb") as f:
                pickle.dump(var_temp_result, f)

            # Xây dựng và huấn luyện mô hình VARNN cuối cùng
            final_model = build_varnn(
                input_dim=X_train_vn.shape[1],
                output_dim=n_features,
                hidden_layer_sizes=best_params["hidden_layer_sizes"],
                learning_rate=best_params["learning_rate"],
            )

            history = train_and_save_model(
                final_model,
                X_train_vn,
                y_train_vn.values,
                X_val_vn,
                y_val_vn.values,
                file_name,
                "VARNN",
                best_params,
                stop_training
            )
            
            if not stop_training.is_set() and history is not None:
                st.write("### Training History")
                plot_training_history(history.history)

        elif model_type == "Hybrid VARNN":
            var_model = VAR(train_data)
            var_result = var_model.fit(maxlags=5, ic="aic")
            best_lags = var_result.k_ar
            train_var_pred = create_var_predictions(train_data, var_result, best_lags, train_data.columns)
            val_var_pred = create_var_predictions(val_data, var_result, best_lags, val_data.columns)

            param_grid = {
                "learning_rate": [0.0001, 0.001, 0.01],
                "batch_size": [16, 32, 64],
                "hidden_layer_sizes": [32, 64, 128],
                "epoch": [50, 100, 150],
            }
            
            best_params, best_mse, search_time = grid_search(
                train_var_pred,
                train_data.values[best_lags:],
                val_var_pred,
                val_data.values[best_lags:],
                param_grid,
                stop_training_flag=stop_training
            )

            st.success("Parameter optimization completed.")
            st.write("**Best Parameters:**")
            st.json(best_params)
            st.write(f"**Best MSE:** {best_mse:.4f}")
            st.write(f"**Search Time:** {search_time:.2f} seconds")
            st.write(f"**Best Lags (VAR):** {best_lags}")

            save_grid_search_results(file_name, model_type, best_params, best_mse, search_time, best_lags)
            with open(f"results/{file_name}/var_result.pkl", "wb") as f:
                pickle.dump(var_result, f)

            final_model = build_varnn(
                input_dim=train_var_pred.shape[1],
                output_dim=train_var_pred.shape[1],
                hidden_layer_sizes=best_params["hidden_layer_sizes"],
                learning_rate=best_params["learning_rate"],
            )

            history = train_and_save_model(
                final_model,
                train_var_pred,
                train_data.values[best_lags:],
                val_var_pred,
                val_data.values[best_lags:],
                file_name,
                model_type,
                best_params,
                stop_training
            )
            
            if not stop_training.is_set() and history is not None:
                st.write("### Training History")
                plot_training_history(history.history)

        elif model_type == "VAR":
            st.subheader("Training VAR Model")
            var_model = VAR(train_data)
            var_result = var_model.fit(maxlags=5, ic="aic")
            best_lags = var_result.k_ar
            st.write(f"VAR Model fitted with optimal lag order: {best_lags}")
            os.makedirs(f"results/{file_name}", exist_ok=True)
            with open(f"results/{file_name}/var_result.pkl", "wb") as f:
                pickle.dump(var_result, f)
            save_grid_search_results(file_name, model_type, {"maxlags": best_lags}, best_lags=best_lags)
            st.success("VAR model training completed and model saved.")
            
    if  stop_training.is_set():
        display_last_saved_results(file_name, model_type)

    if test_button:
        start_time = time.time()
        st.subheader("Testing Model")

        grid_search_path = f"results/{file_name}/{model_type}_grid_search_results.json"
        if not os.path.exists(grid_search_path):
            st.error("Grid search results not found. Please train the model first.")
        else:
            with open(grid_search_path, "r") as f:
                grid_search_results = json.load(f)
        best_lags = grid_search_results.get("best_lags", None)
        if best_lags is None:
            st.error("Best lags not found in grid search results.")
        else:
            if model_type == "VAR":
                try:
                    with open(f"results/{file_name}/var_result.pkl", "rb") as f:
                        var_result = pickle.load(f)
                except FileNotFoundError:
                    st.error("VAR model results not found. Please train the model first.")
                predictions = var_result.forecast(test_data.values[-best_lags:], steps=len(test_data))
                predictions = np.array(predictions)
                ground_truth = test_data
                execution_time = time.time() - start_time
                st.subheader("Actual vs Predicted Values")
                visualize_predictions(ground_truth, predictions, test_data.columns)
                evaluation_df = evaluate_predictions(ground_truth, predictions, execution_time)
                st.subheader("Evaluation")
                st.table(evaluation_df)

            elif model_type == "Hybrid VARNN":
                model_path = f"models/{file_name}/{model_type}_final_model.keras"
                if not os.path.exists(model_path):
                    st.error("Trained model not found. Please train the model first.")
                else:
                    varnn_model = tf.keras.models.load_model(model_path)
                    try:
                        with open(f"results/{file_name}/var_result.pkl", "rb") as f:
                            var_result = pickle.load(f)
                    except FileNotFoundError:
                        st.error("VAR model results not found. Please run optimization first.")
                    test_var_pred = create_var_predictions(test_data, var_result, best_lags, test_data.columns)
                    st.write("Testing predictions created successfully!")
                    predictions = varnn_model.predict(test_var_pred)
                    ground_truth = test_data.iloc[best_lags:]
                    execution_time = time.time() - start_time
                    st.subheader("Actual vs Predicted Values")
                    visualize_predictions(ground_truth, predictions, test_data.columns)
                    evaluation_df = evaluate_predictions(ground_truth, predictions, execution_time)
                    st.subheader("Evaluation")
                    st.table(evaluation_df)

            else:  # VARNN
                model_path = f"models/{file_name}/{model_type}_final_model.keras"
                if not os.path.exists(model_path):
                    st.error("Trained model not found. Please train the model first.")
                else:
                    varnn_model = tf.keras.models.load_model(model_path)
                    try:
                        with open(f"results/{file_name}/var_result.pkl", "rb") as f:
                            var_result = pickle.load(f)
                    except FileNotFoundError:
                        st.error("VAR model results not found. Please run optimization first.")
                    test_lagged = create_lagged_features(test_data, best_lags)
                    n_features = len(test_data.columns)
                    X_test_vn = test_lagged.iloc[:, n_features:]
                    y_test_vn = test_lagged.iloc[:, :n_features]
                    predictions = varnn_model.predict(X_test_vn)
                    predictions = predictions[: len(y_test_vn)]
                    ground_truth = y_test_vn
                    execution_time = time.time() - start_time
                    st.subheader("Actual vs Predicted Values")
                    visualize_predictions(ground_truth, predictions, test_data.columns)
                    evaluation_df = evaluate_predictions(ground_truth, predictions, execution_time)
                    st.subheader("Evaluation")
                    st.table(evaluation_df)

    # Lưu lại các thông tin cần dùng trên các trang khác
    st.session_state['file_name'] = file_name
    st.session_state['data'] = data
    st.session_state['model_type'] = model_type
# Standard libraries
import os
import json
import threading
import time

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Streamlit for UI
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode, JsCode

# Statistical and time series analysis
from statsmodels.tsa.api import VAR

# Machine learning and preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

# Deep learning with TensorFlow
import tensorflow as tf

# Utility libraries
import pickle

# main.py
from preprocessing import (
    preprocess_data,
    split_train_test,
    split_train_val,
    make_stationary,
)
from models import create_var_predictions, build_varnn
from visualization import visualize_predictions, visualize_column
from statistic import check_stationarity
from utils import grid_search, augment_timeseries_data, augment_with_gaussian

# Define function to create lagged features
def create_lagged_features(df, lags):
    df_lagged = df.copy()
    for lag in range(1, lags + 1):
        lagged = df.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
        df_lagged = pd.concat([df_lagged, lagged], axis=1)
    return df_lagged.dropna()
                

def main():
    st.title('Macroeconomic Multivariate Forecasting Tool')
    
    # Step 1: Upload and Display Data
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type='csv', key='upload_train')
    
    if uploaded_file:
        file_name = uploaded_file.name[:-4]
        directory_results = f'results/{file_name}'
        directory_models = f'models/{file_name}'
        
        if not os.path.exists(directory_results):
            os.makedirs(directory_results)
            
        if not os.path.exists(directory_models):
            os.makedirs(directory_models)
            
        data = pd.read_csv(uploaded_file)

        # Step 2: Preprocess data
        data = preprocess_data(data)  
        
        # Step 3: Augment data
        augment_method = st.sidebar.selectbox(
            "Choose Augmentation Method",
            ("Gaussian Noise", "Numpy")
        )
        
        data_aug = data
        aug_box = st.sidebar.checkbox("Augment Data")
        if aug_box:
            if augment_method == "Gaussian Noise":                        
                data_aug = augment_with_gaussian(data)
            elif augment_method == "Numpy":            
                data_aug = augment_timeseries_data(data)
                
        data = data_aug
        st.subheader("Raw Data")
        st.dataframe(data)

        selected_col = st.sidebar.selectbox("Select Column to Visualize", data.columns)
        visualize_column(data, selected_col)
        
        # Step 4: Stationary check
        if st.sidebar.checkbox("Check Stationarity"):
            stationarity_results = check_stationarity(data)
            st.subheader("Stationarity Check Results")
            st.dataframe(stationarity_results)

        # Step 5: Make data stationary if needed
        if st.sidebar.checkbox("Make Data Stationary"):
            data = make_stationary(data)
            st.subheader("Stationary Data")
            st.dataframe(data)
            
            visualize_column(data, selected_col, description="(After making stationary)")
            
        # Step 6: Normalization
        normalization_method = st.sidebar.radio("Select Data Normalization Method", ["No Normalization", "Z-Score Normalization", "MinMax Normalization"])
        if normalization_method == "MinMax Normalization":
            val_input = st.sidebar.text_input("Enter (minimum, maximum) values", "0,1")
            min_val, max_val = map(int, val_input.split(","))

        if normalization_method == "Z-Score Normalization":
            scaler = StandardScaler()
        elif normalization_method == "MinMax Normalization":
            scaler = MinMaxScaler(feature_range=(min_val, max_val))
        else:
            scaler = None

        if scaler:
            scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
        else:
            scaled_data = data

        data = scaled_data
        st.subheader(f"Processed Data ({normalization_method})")
        st.dataframe(data)
        visualize_column(data, selected_col, description="(After normalization)")

        # Step 7: Train Test Val Split
        st.sidebar.header("Data Splitting")
        train_test_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.9, 0.8)
        test_ratio = 1 - train_test_ratio
        st.sidebar.text(f"Train: {train_test_ratio:.1f}, Test: {test_ratio:.1f}")
        train_data, test_data = split_train_test(data, train_test_ratio)

        st.write(f"Train-Test Split: Train: {len(train_data)} rows, Test: {len(test_data)} rows")

        # Train-Validation Split Ratio (within Train)
        train_val_ratio = st.sidebar.slider("Train-Validation Split Ratio (within Train Data)", 0.1, 0.9, 0.2, key="train_val_slider")
        val_ratio = 1 - train_val_ratio
        st.sidebar.text(f"Validation: {train_val_ratio:.1f}, Train: {val_ratio:.1f}")
        train_data_final, val_data = split_train_val(train_data, train_val_ratio)

        if len(train_data_final) == 0 or len(val_data) == 0:
            st.error("Train-Validation ratio is invalid. Adjust the slider!")
        else:
            st.write(f"Train-Validation Split: Train: {len(train_data_final)} rows, Validation: {len(val_data)} rows")

        # Step 8: Train Model  
        model_type = st.sidebar.selectbox("Select Model", ["VARNN", "Hybrid VARNN", "VAR"])

        train_button = st.sidebar.button("Train Model")
        stop_button = st.sidebar.button("Stop Training")
        test_button = st.sidebar.button("Test Model")
        predict_button = st.sidebar.button("Predict Future")

        stop_training = threading.Event()

        # Callback to Stop Training
        class StopTrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if stop_training.is_set():
                    self.model.stop_training = True
                    st.warning(f"Training stopped at epoch {epoch + 1}")

        if stop_button:
            stop_training.set()
            st.warning("Stopping training process...")
            
        global flag
        flag = False
        
        if train_button:
            st.subheader("Parameter Optimization and Training")
            stop_training.clear()
            
            # ================================
            # VARNN Model Implementation
            # ================================
            if model_type == "VARNN":
                st.subheader("Training VARNN Model")
                # Determine optimal lag order using VAR on training data
                var_temp = VAR(train_data)
                var_temp_result = var_temp.fit(maxlags=5, ic='aic')
                best_lags = var_temp_result.k_ar
                st.write(f"Optimal lag order determined by VAR: {best_lags}")
                
                # Create lagged datasets
                train_lagged = create_lagged_features(train_data, best_lags)
                val_lagged = create_lagged_features(val_data, best_lags)
                test_lagged = create_lagged_features(test_data, best_lags)
                
                n_features = len(train_data.columns)
                X_train_vn = train_lagged.iloc[:, n_features:]
                y_train_vn = train_lagged.iloc[:, :n_features]
                X_val_vn = val_lagged.iloc[:, n_features:]
                y_val_vn = val_lagged.iloc[:, :n_features]
                
                # Grid search for best hyperparameters
                param_grid = {
                    'learning_rate': [0.0001, 0.001, 0.01],
                    'batch_size': [16, 32, 64],
                    'hidden_layer_sizes': [32, 64, 128],
                    'epoch': [50, 100, 150]
                }
                
                best_params, best_mse, search_time = grid_search(
                    X_train_vn,
                    y_train_vn.values,
                    X_val_vn,
                    y_val_vn.values,
                    param_grid,
                )
                
                st.success("Parameter optimization completed for VARNN.")
                st.write("**Best Parameters:**")
                st.json(best_params)
                st.write(f"**Best MSE:** {best_mse:.4f}")
                st.write(f"**Search Time:** {search_time:.2f} seconds")
                st.write(f"**Best Lags (VAR):** {best_lags}")
                
                # Save grid search results
                grid_search_results = {
                    'model_type': model_type,
                    'best_parameters': best_params,
                    'best_mse': best_mse,
                    'search_time': search_time,
                    'best_lags': best_lags
                }
                with open(f'results/{file_name}/{model_type}_grid_search_results.json', 'w') as f:
                    json.dump(grid_search_results, f)
                
                # Save the VAR model result (for lag selection later)
                with open(f'results/{file_name}/var_result.pkl', 'wb') as f:
                    pickle.dump(var_temp_result, f)
                
                # Build and train final VARNN model
                final_model = build_varnn(
                    input_dim=X_train_vn.shape[1],
                    output_dim=n_features,
                    hidden_layer_sizes=best_params['hidden_layer_sizes'],
                    learning_rate=best_params['learning_rate'],
                )
                
                try:
                    history = final_model.fit(
                        X_train_vn,
                        y_train_vn.values,
                        validation_data=(X_val_vn, y_val_vn.values),
                        epochs=best_params['epoch'],
                        batch_size=best_params['batch_size'],
                        verbose=1,
                        callbacks=[
                            StopTrainingCallback(),
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=10,
                                restore_best_weights=True
                            )
                        ]
                    )
                    
                    with open(f'results/{file_name}/training_history.pkl', "wb") as f:
                        pickle.dump(history.history, f)
                        
                    if not stop_training.is_set():
                        final_model.save(f'models/{file_name}/VARNN_final_model.keras')
                        st.success("VARNN training completed and model saved.")
                        
                except Exception as e:
                    st.error(f"Training stopped or failed: {e}")

            
            # ================================
            # Hybrid VARNN Model (already implemented)
            # ================================
            if model_type == "Hybrid VARNN":
                var_model = VAR(train_data)
                var_result = var_model.fit(maxlags=5, ic='aic')
                best_lags = var_result.k_ar
                train_var_pred = create_var_predictions(train_data, var_result, var_result.k_ar, data.columns)
                val_var_pred = create_var_predictions(val_data, var_result, var_result.k_ar, data.columns)

                param_grid = {
                    'learning_rate': [0.0001, 0.001, 0.01],
                    'batch_size': [16, 32, 64],
                    'hidden_layer_sizes': [32, 64, 128],
                    'epoch': [50, 100, 150]
                }

                best_params, best_mse, search_time = grid_search(
                    train_var_pred,
                    train_data.values[var_result.k_ar:],
                    val_var_pred,
                    val_data.values[var_result.k_ar:],
                    param_grid,
                )

                st.success("Parameter optimization completed.")
                st.write("**Best Parameters:**")
                st.json(best_params)
                st.write(f"**Best MSE:** {best_mse:.4f}")
                st.write(f"**Search Time:** {search_time:.2f} seconds")
                st.write(f"**Best Lags (VAR):** {best_lags}")

                with open(f'results/{file_name}/var_result.pkl', 'wb') as f:
                    pickle.dump(var_result, f)

                grid_search_results = {
                    'model_type': model_type,
                    'best_parameters': best_params,
                    'best_mse': best_mse,
                    'search_time': search_time,
                    'best_lags': best_lags
                }

                with open(f'results/{file_name}/{model_type}_grid_search_results.json', 'w') as f:
                    json.dump(grid_search_results, f)

                final_model = build_varnn(
                    input_dim=train_var_pred.shape[1],
                    output_dim=train_var_pred.shape[1],
                    hidden_layer_sizes=best_params['hidden_layer_sizes'],
                    learning_rate=best_params['learning_rate'],
                )

                try:
                    history = final_model.fit(
                        train_var_pred,
                        train_data.values[best_lags:],
                        validation_data=(val_var_pred, val_data.values[best_lags:]),
                        epochs=best_params['epoch'],
                        batch_size=best_params['batch_size'],
                        verbose=1,
                        callbacks=[StopTrainingCallback(), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
                    )
                    
                    with open(f'results/{file_name}/training_history.pkl', "wb") as f:
                        pickle.dump(history.history, f)
                        
                    if not stop_training.is_set():
                        final_model.save(f'models/{file_name}/{model_type}_final_model.keras')
                        st.success("Training completed and model saved.")
                        
                except Exception as e:
                    st.error(f"Training stopped or failed: {e}")

                if stop_training.is_set():
                    st.warning("Training was stopped by the user.")
                else:
                    st.write("### Training History")
                    loss_df = pd.DataFrame({
                        'Epoch': range(1, len(history.history['loss']) + 1),
                        'Training Loss': history.history['loss'],
                        'Validation Loss': history.history['val_loss']
                    })
                    st.line_chart(loss_df.set_index('Epoch'))

                    st.write("### Final Losses")
                    st.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
                    st.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
                    
            # ================================
            # VAR Model Implementation
            # ================================
            if model_type == "VAR":
                st.subheader("Training VAR Model")
                var_model = VAR(train_data)
                var_result = var_model.fit(maxlags=5, ic='aic')
                best_lags = var_result.k_ar
                st.write(f"VAR Model fitted with optimal lag order: {best_lags}")

                # Ensure directory exists
                os.makedirs(f'results/{file_name}', exist_ok=True)
                
                # Save the fitted VAR model result
                with open(f'results/{file_name}/var_result.pkl', 'wb') as f:
                    pickle.dump(var_result, f)

                # Save grid search results (for consistency, even though no search was performed)
                grid_search_results = {
                    'model_type': model_type,
                    'best_parameters': {"maxlags": best_lags},
                    'best_lags': best_lags
                }
                with open(f'results/{file_name}/{model_type}_grid_search_results.json', 'w') as f:
                    json.dump(grid_search_results, f)

                st.success("VAR model training completed and model saved.")

                
        # Step 9: Test Model                
        if test_button:
            start_time = time.time()
            st.subheader("Testing Model")

            # Verify that grid search results exist
            grid_search_path = f'results/{file_name}/{model_type}_grid_search_results.json'
            if not os.path.exists(grid_search_path):
                st.error("Grid search results not found. Please train the model first.")
                return
            else:
                with open(grid_search_path, 'r') as f:
                    grid_search_results = json.load(f)

            best_lags = grid_search_results.get('best_lags', None)
            if best_lags is None:
                st.error("Best lags not found in grid search results.")
                return

            # Depending on the model type, use different testing logic
            if model_type == "VAR":
                # For the VAR model, we don't have a Keras model.
                try:
                    with open(f'results/{file_name}/var_result.pkl', 'rb') as f:
                        var_result = pickle.load(f)
                except FileNotFoundError:
                    st.error("VAR model results not found. Please train the model first.")
                    return

                # Create predictions using the VAR model directly.
                # Note: Here we forecast for the entire test set.
                predictions = var_result.forecast(test_data.values[-best_lags:], steps=len(test_data))
                predictions = np.array(predictions)
                ground_truth = test_data  # Use the full test set
                execution_time = time.time() - start_time
                st.subheader("Actual vs Predicted Values")
                visualize_predictions(ground_truth, predictions, test_data.columns)

                # Evaluation metrics
                test_mse = mean_squared_error(ground_truth, predictions)
                test_mae = mean_absolute_error(ground_truth, predictions)
                test_rmse = np.sqrt(test_mse)
                y_mean = np.mean(ground_truth)
                cv_rmse = test_rmse / y_mean

                evaluation_metrics = {
                    "Metric": ["Test Time (seconds)", "Test MSE", "Test MAE", "Test RMSE", "Test CV RMSE"],
                    "Value": [f"{execution_time:.2f}", f"{test_mse:.4f}", f"{test_mae:.4f}", f"{test_rmse:.4f}", f"{cv_rmse:.4f}"]
                }

                evaluation_df = pd.DataFrame(evaluation_metrics)
                st.subheader("Evaluation")
                st.table(evaluation_df)
                
            elif model_type == "Hybrid VARNN":
                # For Hybrid VARNN, load the Keras model.
                model_path = f'models/{file_name}/{model_type}_final_model.keras'
                if not os.path.exists(model_path):
                    st.error("Trained model not found. Please train the model first.")
                    return

                varnn_model = tf.keras.models.load_model(model_path)

                try:
                    with open(f'results/{file_name}/var_result.pkl', 'rb') as f:
                        var_result = pickle.load(f)
                except FileNotFoundError:
                    st.error("VAR model results not found. Please run optimization first.")
                    return

                # Create lagged predictions for the test set
                test_var_pred = create_var_predictions(test_data, var_result, best_lags, test_data.columns)
                st.write("Testing predictions created successfully!")
                        
                # Get predictions from the trained neural network
                predictions = varnn_model.predict(test_var_pred)
                # Align ground truth: test_var_pred is created starting from index best_lags
                ground_truth = test_data.iloc[best_lags:]

                           # Get predictions from the trained neural network

                execution_time = time.time() - start_time
                st.subheader("Actual vs Predicted Values")
                visualize_predictions(ground_truth, predictions, test_data.columns)

                # Evaluation metrics
                test_mse = mean_squared_error(ground_truth, predictions)
                test_mae = mean_absolute_error(ground_truth, predictions)
                test_rmse = np.sqrt(test_mse)
                y_mean = np.mean(ground_truth)
                cv_rmse = test_rmse / y_mean

                evaluation_metrics = {
                    "Metric": ["Test Time (seconds)", "Test MSE", "Test MAE", "Test RMSE", "Test CV RMSE"],
                    "Value": [f"{execution_time:.2f}", f"{test_mse:.4f}", f"{test_mae:.4f}", f"{test_rmse:.4f}", f"{cv_rmse:.4f}"]
                }

                evaluation_df = pd.DataFrame(evaluation_metrics)
                st.subheader("Evaluation")
                st.table(evaluation_df)
            
            else:
                # Load the trained VARNN model
                model_path = f'models/{file_name}/{model_type}_final_model.keras'
                if not os.path.exists(model_path):
                    st.error("Trained model not found. Please train the model first.")
                    return

                varnn_model = tf.keras.models.load_model(model_path)

                try:
                    with open(f'results/{file_name}/var_result.pkl', 'rb') as f:
                        var_result = pickle.load(f)
                except FileNotFoundError:
                    st.error("VAR model results not found. Please run optimization first.")
                    return

                # Create lagged features for test data
                test_lagged = create_lagged_features(test_data, best_lags)

                # Extract input (X) and ground truth (y) directly from the lagged data
                n_features = len(test_data.columns)
                X_test_vn = test_lagged.iloc[:, n_features:]
                y_test_vn = test_lagged.iloc[:, :n_features]

                
                # Print shapes to debug
                print("X_test_vn shape:", X_test_vn.shape)
                print("Expected input shape:", varnn_model.input_shape)

                # Make predictions
                predictions = varnn_model.predict(X_test_vn)
                predictions = predictions[:len(y_test_vn)]  # Ensure same length

                ground_truth = y_test_vn


                # Get predictions from the trained neural network

                execution_time = time.time() - start_time
                st.subheader("Actual vs Predicted Values")
                visualize_predictions(ground_truth, predictions, test_data.columns)

                # Evaluation metrics
                test_mse = mean_squared_error(ground_truth, predictions)
                test_mae = mean_absolute_error(ground_truth, predictions)
                test_rmse = np.sqrt(test_mse)
                y_mean = np.mean(ground_truth)
                cv_rmse = test_rmse / y_mean

                evaluation_metrics = {
                    "Metric": ["Test Time (seconds)", "Test MSE", "Test MAE", "Test RMSE", "Test CV RMSE"],
                    "Value": [f"{execution_time:.2f}", f"{test_mse:.4f}", f"{test_mae:.4f}", f"{test_rmse:.4f}", f"{cv_rmse:.4f}"]
                }

                evaluation_df = pd.DataFrame(evaluation_metrics)
                st.subheader("Evaluation")
                st.table(evaluation_df)


            
        # Step 10: Predict Future Values
        st.subheader("Predict Future Values") 
        file_path = f'results/{file_name}/{model_type}_grid_search_results.json'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                grid_search_results = json.load(f)
            best_lags = grid_search_results.get('best_lags', None)
        else:
            st.write("**Please train to predict future!**")
            best_lags = 0           
        last_date = data.index.max().year
        list_year = [str(i) for i in range(last_date+1,last_date+best_lags+1)]
        data_pre = {"Series Name": list_year}
        for x in data.columns:
            data_pre[x] = [0 for _ in range(len(list_year))]
        df_pre = pd.DataFrame(data=data_pre)

        gd = GridOptionsBuilder.from_dataframe(df_pre)
        gd.configure_default_column(editable=True)
        gridoptions = gd.build()

        with st.form('Inventory') as f:
            st.header('A list of values is needed 🔖')
            response = AgGrid(df_pre,
                            gridOptions = gridoptions, 
                            editable=True,
                            allow_unsafe_jscode = True,
                            theme = 'balham',
                            height = 200,
                            fit_columns_on_grid_load = True)
            st.write(" *Note: Please fill this to predict next step.*")
            st.form_submit_button("Confirm item(s) 🔒", type="primary")
        st.subheader("Updated List")
        st.write(response['data']) 
        data_predict = pd.DataFrame(response['data'])
        
        if predict_button:
            st.subheader("Predict Future Values")
            data_predict['Date'] = pd.to_datetime(data_predict['Series Name'], format='%Y') 
            data_predict.set_index('Date', inplace=True)
            data_predict.drop('Series Name', axis=1, inplace=True)
            try:
                model_path = f'models/{file_name}/{model_type}_final_model.keras'
                if not os.path.exists(model_path):
                    st.error("Trained model not found. Please train the model first.")
                    return
                varnn_model = tf.keras.models.load_model(model_path)

                with open(f'results/{file_name}/var_result.pkl', 'rb') as f:
                    var_result = pickle.load(f)
                features = data_predict.columns
                len_data = len(data_predict)
                var_predict = var_result.forecast(data_predict[features].values[0:len_data], steps=1)
                varnn_predict = varnn_model.predict(var_predict)

                varnn_predict = np.array(varnn_predict)
                last_date = data_predict.index[-1]
                future_dates = pd.date_range(start=last_date, periods=2, freq='YS')[1:]
                future_df = pd.DataFrame(varnn_predict, columns=data_predict.columns, index=future_dates)

                st.subheader("Future Predictions")
                st.dataframe(future_df)

                csv = future_df.to_csv(index=True)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="future_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()

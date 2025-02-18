# Standard libraries
import os
import json
import pickle

# Data manipulation and visualization
import pandas as pd
import numpy as np

# Streamlit for UI and AgGrid for bảng dữ liệu có thể chỉnh sửa
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder

# Statistical and time series analysis
from statsmodels.tsa.api import VAR

# Deep learning with TensorFlow
import tensorflow as tf

def show_predict_page():
    st.title("Predict Future Values")
    
    # Kiểm tra xem các thông tin cần thiết đã có (được lưu vào session_state) hay chưa
    if 'file_name' not in st.session_state or st.session_state['file_name'] is None:
        st.error("No trained model found. Please train the model first on the Home page.")
        return
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Data not found. Please upload data on the Home page.")
        return
    if 'model_type' not in st.session_state or st.session_state['model_type'] is None:
        st.error("Model type not found. Please train the model first on the Home page.")
        return

    file_name = st.session_state['file_name']
    data = st.session_state['data']
    model_type = st.session_state['model_type']
    
    # Lấy giá trị best_lags từ kết quả grid search đã lưu
    file_path = f'results/{file_name}/{model_type}_grid_search_results.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            grid_search_results = json.load(f)
        best_lags = grid_search_results.get('best_lags', None)
    else:
        st.write("**Please train to predict future!**")
        best_lags = 0

    if best_lags is None or best_lags == 0:
        st.error("Invalid best lags. Please train your model first on the Home page.")
        return

    # Tạo bảng nhập dữ liệu cho dự báo (sử dụng AgGrid)
    last_date = data.index.max().year
    list_year = [str(i) for i in range(last_date + 1, last_date + best_lags + 1)]
    data_pre = {"Series Name": list_year}
    for col in data.columns:
        data_pre[col] = [0] * len(list_year)
    df_pre = pd.DataFrame(data=data_pre)

    gd = GridOptionsBuilder.from_dataframe(df_pre)
    gd.configure_default_column(editable=True)
    gridoptions = gd.build()

    with st.form('predict_form'):
        st.header('Enter Future Values 🔖')
        response = AgGrid(
            df_pre,
            gridOptions=gridoptions, 
            editable=True,
            allow_unsafe_jscode=True,
            theme='balham',
            height=200,
            fit_columns_on_grid_load=True
        )
        st.write(" *Note: Please fill in this table to predict the next step.*")
        submitted = st.form_submit_button("Confirm")

    st.subheader("Updated List")
    st.write(response['data'])
    data_predict = pd.DataFrame(response['data'])

    if submitted:
        st.subheader("Predict Future Values")
        # Chuyển cột 'Series Name' thành index dạng datetime
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
            # Tạo dự báo từ VAR
            var_predict = var_result.forecast(data_predict[features].values[:len_data], steps=1)
            # Dự báo tiếp qua mô hình varnn
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
import os

import pandas as pd

import streamlit as st

from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Custom module imports
from helpers.preprocessing import preprocess_data, split_train_test, split_train_val, make_stationary
from helpers.visualization import visualize_column
from helpers.statistic import check_stationarity
from helpers.utils import augment_with_numpy, augment_with_gaussian

def create_directories(file_name: str) -> tuple:
    """
    Create directories for saving results and models.
    
    Args:
        file_name (str): Name of the dataset file (without extension).
    
    Returns:
        tuple: Paths to results and models directories.
    """
    directory_results = os.path.join('results', file_name)
    directory_models = os.path.join('models', file_name)
    os.makedirs(directory_results, exist_ok=True)
    os.makedirs(directory_models, exist_ok=True)
    return directory_results, directory_models

def upload_data() -> tuple:
    """
    Handle CSV file upload via Streamlit sidebar.
    
    Returns:
        tuple: DataFrame containing uploaded data and the file name.
    """
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type='csv', key='upload_train')
    if not uploaded_file:
        return None, None
    
    file_name = os.path.splitext(uploaded_file.name)[0]
    data = pd.read_csv(uploaded_file)
    return data, file_name

def apply_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the given DataFrame."""
    return preprocess_data(data)

def augment_data(data: pd.DataFrame, selected_col: str) -> pd.DataFrame:
    """
    Augment data based on user selection.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: Augmented DataFrame.
    """
    if st.sidebar.checkbox("Augment Data"):
        method = st.sidebar.selectbox("Choose Augmentation Method", ("Gaussian Noise", "Numpy"))
        data = augment_with_gaussian(data) if method == "Gaussian Noise" else augment_with_numpy(data)
        st.subheader("Raw Data After Augmentation")
        st.dataframe(data)
        visualize_column(data, selected_col, description="(After augmentation)")
    return data

def stationarity_checks(data: pd.DataFrame, selected_col: str) -> pd.DataFrame:
    """
    Perform stationarity checks and transformations.
    
    Args:
        data (pd.DataFrame): Input data.
        selected_col (str): Column to visualize after stationarity transformation.
    
    Returns:
        pd.DataFrame: Stationary data if transformation is applied.
    """
    if st.sidebar.checkbox("Check Stationarity"):
        st.subheader("Stationarity Check Results")
        st.dataframe(check_stationarity(data))
    
    if st.sidebar.checkbox("Make Data Stationary"):
        data = make_stationary(data)
        st.subheader("Stationary Data")
        st.dataframe(data)
        visualize_column(data, selected_col, description="(After making stationary)")
    
    return data

def normalize_data(data: pd.DataFrame, selected_col: str) -> pd.DataFrame:
    """
    Normalize the data using the selected method.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
    
    Returns:
        tuple: Normalized DataFrame and the selected method.
    """
    normalization_method = st.sidebar.radio("Select Data Normalization Method", 
                                          ["No Normalization", "Z-Score Normalization", "MinMax Normalization"])
    
    if normalization_method == "MinMax Normalization":
        min_val, max_val = map(int, st.sidebar.text_input("Enter (minimum, maximum) values", "0,1").split(","))
        scaler = MinMaxScaler(feature_range=(min_val, max_val))
        st.subheader(f"Processed Data After MinMax Normalization")
        st.dataframe(data)
        visualize_column(data, selected_col, description="(After normalization)")
        
    elif normalization_method == "Z-Score Normalization":
        scaler = StandardScaler()
        st.subheader(f"Processed Data After Z-Score Normalization")
        st.dataframe(data)
        visualize_column(data, selected_col, description="(After normalization)")
    else:
        return data
    
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

def split_data(data: pd.DataFrame) -> tuple:
    """
    Split data into training, validation, and test sets.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
    
    Returns:
        tuple: Training, validation, and test sets.
    """
    train_test_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.9, 0.8)
    train_data, test_data = split_train_test(data, train_test_ratio)
    
    train_val_ratio = st.sidebar.slider("Train-Validation Split Ratio (within Train Data)", 0.1, 0.9, 0.2, key="train_val_slider")
    train_data_final, val_data = split_train_val(train_data, train_val_ratio)
    
    st.write(f"Train-Test-Validation Ratios: Train: {train_test_ratio}, Test: {1 - train_test_ratio}, Validation: {train_val_ratio}")
    
    if not train_data_final.empty and not val_data.empty:
        st.write(f"Train-Validation Split: Train: {len(train_data_final)} rows, Validation: {len(val_data)} rows")
    else:
        st.error("Train-Validation ratio is invalid. Adjust the slider!")
    
    return train_data_final, val_data, test_data
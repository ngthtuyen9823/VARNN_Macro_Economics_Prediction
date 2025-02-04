# Data manipulation and visualization
import pandas as pd
import numpy as np


def preprocess_data(data, drop_threshold=1):
    data['Year'] = data['Series Name'].str.extract(r'\[YR(\d{4})\]')[0]
    data['Date'] = pd.to_datetime(data['Year'], format='%Y') 
    data.set_index('Date', inplace=True)
    data.drop('Series Name', axis=1, inplace=True)
    data.drop('Year', axis=1, inplace=True)
    data.replace('..', np.nan, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    missing_ratios = data.isnull().mean()
    columns_to_drop = missing_ratios[missing_ratios > drop_threshold].index
    data.drop(columns=columns_to_drop, inplace=True)
    data.interpolate(method='linear', inplace=True, limit_direction='both')
    data = data[~data.index.isnull()]
    data.drop_duplicates(inplace=True)

    return data


def split_train_test(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    return train_data, test_data


def split_train_val(train_data, val_ratio):
    val_size = int(len(train_data) * val_ratio)
    val_data = train_data.iloc[:val_size]
    train_data = train_data.iloc[val_size:]
    
    return train_data, val_data


def difference_series(series, lag=2):
    return series.diff(periods=lag).dropna()


def make_stationary(data, lag=2):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    if not non_numeric_columns.empty:
        print(f"Warning: Non-numeric columns excluded: {list(non_numeric_columns)}")

    numeric_data = data.select_dtypes(include=['number'])
    differenced_data = numeric_data.apply(difference_series, lag=lag)

    return differenced_data
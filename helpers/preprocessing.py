import pandas as pd
import numpy as np

def preprocess_data(data, drop_threshold=1.0):
    """Preprocesses the dataset by extracting year information, handling missing values, 
    converting data types, and performing interpolation.

    Args:
        data (pd.DataFrame): Input dataset containing a 'Series Name' column.
        drop_threshold (float, optional): Fraction of missing values allowed per column 
                                          before dropping it. Defaults to 1.0.

    Returns:
        pd.DataFrame: Cleaned and preprocessed dataset.
    """
    data = data.copy()  # Avoid modifying the original DataFrame

    # Extract the year from 'Series Name' and convert it to a datetime index
    data['Year'] = data['Series Name'].str.extract(r'\[YR(\d{4})\]')[0]
    data['Date'] = pd.to_datetime(data['Year'], format='%Y')
    data.set_index('Date', inplace=True)
    data.drop(columns=['Series Name', 'Year'], inplace=True)

    # Replace missing value indicators and convert all columns to numeric
    data.replace('..', np.nan, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')

    # Drop columns with excessive missing values
    missing_ratios = data.isnull().mean()
    data.drop(columns=missing_ratios[missing_ratios > drop_threshold].index, inplace=True)

    # Interpolate missing values linearly
    data.interpolate(method='linear', inplace=True, limit_direction='both')

    # Remove any remaining rows with NaN indices and drop duplicates
    data = data[~data.index.isnull()]
    data.drop_duplicates(inplace=True)

    return data


def split_train_test(data, train_ratio=0.8):
    """Splits data into training and testing sets.

    Args:
        data (pd.DataFrame): Input dataset with a datetime index.
        train_ratio (float, optional): Proportion of data to allocate for training. Defaults to 0.8.

    Returns:
        tuple: (train_data, test_data)
    """
    train_size = int(len(data) * train_ratio)
    return data.iloc[:train_size], data.iloc[train_size:]


def split_train_val(train_data, val_ratio=0.2):
    """Splits training data into training and validation sets.

    Args:
        train_data (pd.DataFrame): The training dataset.
        val_ratio (float, optional): Proportion of training data to allocate for validation. Defaults to 0.2.

    Returns:
        tuple: (train_data, val_data)
    """
    val_size = int(len(train_data) * val_ratio)
    return train_data.iloc[val_size:], train_data.iloc[:val_size]


def difference_series(series, lag=2):
    """Computes the differenced time series with a specified lag.

    Args:
        series (pd.Series): Input time series.
        lag (int, optional): Lag for differencing. Defaults to 2.

    Returns:
        pd.Series: Differenced series with NaN values dropped.
    """
    return series.diff(periods=lag).dropna()


def make_stationary(data, lag=2):
    """Transforms time series data to make it stationary by applying differencing.

    Args:
        data (pd.DataFrame): Input dataset containing numeric time series columns.
        lag (int, optional): Lag for differencing. Defaults to 2.

    Returns:
        pd.DataFrame: Differenced dataset with only numeric columns.
    
    Raises:
        ValueError: If the input data is not a DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # Identify and exclude non-numeric columns
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    if not non_numeric_columns.empty:
        print(f"Warning: Excluding non-numeric columns: {list(non_numeric_columns)}")

    return data.select_dtypes(include=['number']).apply(difference_series, lag=lag)

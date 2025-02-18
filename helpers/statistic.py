import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(series, column_name=''):
    """Performs the Augmented Dickey-Fuller (ADF) test for stationarity on a given time series.

    Args:
        series (pd.Series): The time series to be tested.
        column_name (str, optional): The name of the column being tested (for reference in results).

    Returns:
        dict: A dictionary containing the ADF test statistic, p-value, and stationarity status.
    """
    series_clean = series.dropna()  # Remove missing values
    result = adfuller(series_clean)  # Perform ADF test

    return {
        'Column': column_name,
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Stationary': 'Yes' if result[1] < 0.05 else 'No'
    }

def check_stationarity(data):
    """Checks stationarity for all columns in a DataFrame using the ADF test.

    Args:
        data (pd.DataFrame): The dataset containing time series columns.

    Returns:
        pd.DataFrame: A DataFrame summarizing ADF test results for each column.
    """
    results = [adf_test(data[column], column_name=column) for column in data.columns]
    return pd.DataFrame(results)
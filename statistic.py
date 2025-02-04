# Data manipulation and visualization
import pandas as pd

# Statistical and time series analysis
from statsmodels.tsa.stattools import adfuller


def adf_test_varnn(series, title=''):
    series_clean = series.dropna()
    result = adfuller(series_clean)
    
    return {
        'Column': title,
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Stationary': 'Yes' if result[1] < 0.05 else 'No'
    }

def check_stationarity(data):
    results = []
    
    for column in data.columns:
        result = adf_test_varnn(data[column], title=column)
        results.append(result)

    return pd.DataFrame(results)


import pandas as pd
import numpy as np


def add_gaussian_noise(time_series, mean=0.0, stddev=0.05):
    """Adds Gaussian noise to a given time series.

    Args:
        time_series (pd.Series): The original time series data.
        mean (float, optional): Mean of the Gaussian noise. Defaults to 0.0.
        stddev (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.05.

    Returns:
        pd.Series: Time series with added Gaussian noise.
    """
    noise = np.random.normal(mean, stddev, len(time_series))
    return time_series + noise


def generate_new_dates(df, periods):
    """Generates a new date range ending one year before the earliest date in the DataFrame index.

    Args:
        df (pd.DataFrame): DataFrame containing a datetime index.
        periods (int): Number of periods to generate.

    Returns:
        pd.DatetimeIndex: A new date range.
    """
    first_date = df.index.min()
    return pd.date_range(end=first_date - pd.DateOffset(years=1), periods=periods, freq='YS')


def augment_with_gaussian(data, mean=0.0, stddev=0.05):
    """Augments a dataset by adding Gaussian noise to numeric columns and generating new timestamps.

    Args:
        data (pd.DataFrame): Original dataset with a datetime index.
        mean (float, optional): Mean of the Gaussian noise. Defaults to 0.0.
        stddev (float, optional): Standard deviation of the Gaussian noise. Defaults to 0.05.

    Returns:
        pd.DataFrame: The original dataset combined with the augmented version.
    """
    augmented_data = data.copy()

    for column in data.select_dtypes(include=[np.number]).columns:
        augmented_data[column] = add_gaussian_noise(data[column].dropna(), mean, stddev)

    new_dates = generate_new_dates(data, len(data))
    augmented_data.index = new_dates

    return pd.concat([data, augmented_data], axis=0).sort_index()


def augment_with_numpy(data, n_periods=24):
    """Augments the dataset using trend, seasonality, and noise for each numeric column.

    Args:
        data (pd.DataFrame): Original dataset with a datetime index.
        n_periods (int, optional): Number of periods to generate. Defaults to 24.

    Returns:
        pd.DataFrame: The original dataset combined with the generated synthetic data.
    """
    new_dates = generate_new_dates(data, n_periods)
    augmented_data = {}

    for column in data.select_dtypes(include=[np.number]).columns:
        mean = data[column].mean()
        std = data[column].std()
        trend = np.polyfit(range(len(data)), data[column].values, 1)[0]

        base_trend = -np.arange(n_periods)[::-1] * trend + mean
        seasonality = np.sin(np.linspace(0, 2 * np.pi, 12)) * std * 0.5
        noise = np.random.normal(0, std * 0.1, n_periods)

        new_series = pd.Series(
            base_trend + np.tile(seasonality, (n_periods // 12) + 1)[:n_periods] + noise,
            index=new_dates
        )
        augmented_data[column] = new_series

    augmented_df = pd.DataFrame(augmented_data)
    return pd.concat([augmented_df, data]).sort_index()

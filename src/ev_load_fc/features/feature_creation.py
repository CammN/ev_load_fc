import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calender
from typing import Literal
import logging
logger = logging.getLogger(__name__)



def aggregate_features(df:pd.DataFrame, out_name:str, substr1:str, substr2:str='') -> pd.DataFrame:
    """
    Aggregates a set of columns row-wise using summation.

    Args:
        df (pd.DataFrame): Input DataFrame.
        out_name (str): Name of aggregated column.
        substr1 (str): Primary substring to identify columns in df to aggregate.
        substr2 (str, optional): Secondary substring to identify columns in df to aggregate. Defaults ''

    Returns:
        pd.DataFrame: Input DataFrame + aggregated column
    """
    df_agg = df.copy()

    fname_cols = [col for col in df_agg.columns if (substr1 in col) and (substr2 in col)]
    
    df_agg[out_name] = 0
    for col in fname_cols:
        df_agg[out_name] += df_agg[col]

    return df_agg


def time_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    Creates time-based sinusoidal features (month, weekday and hour of day) to encode seasonality of a time series.

    Args:
        df (pd.DataFrame): Input DataFrame 

    Returns:
        pd.DataFrame: Input DataFrame + time based features
    """
    df_time = df.copy()

    df_time['month_sin'] = np.sin(2 * np.pi * df_time.index.month/12) 
    df_time['month_cos'] = np.cos(2 * np.pi * df_time.index.month/12) 

    df_time['weekday_sin'] = np.sin(2 * np.pi * df_time.index.weekday/7) 
    df_time['weekday_cos'] = np.cos(2 * np.pi * df_time.index.weekday/7) 

    df_time['hour_sin'] = np.sin(2 * np.pi * df_time.index.hour/24) 
    df_time['hour_cos'] = np.cos(2 * np.pi * df_time.index.hour/24)

    return df_time


def get_holidays(min_timestamp:datetime, max_timestamp:datetime) -> pd.DataFrame:
    """
    Create lookup table for US holidays in a given time period.

    Args:
        min_timestamp (datetime): Minimum timestamp of the time period.
        max_timestamp (datetime): Maximum timestamp of the time period.

    Returns:
        pd.DataFrame: Contains two columns:
                        - 'ds' : the timestamp of the holiday
                        - 'holiday' : the name of the holiday
    """
    # Initialise US holidays data
    cal = calender()
    holidays = cal.holidays(start=min_timestamp, end=max_timestamp, return_name=True)
    holidays_df = holidays.reset_index().rename(columns={'index':'ds', 0:'holiday'})

    # Add extra holidays
    extra_hols ={
        'Christmas Eve': (24,12),
        'Boxing Day': (26,12),
    }
    for hol,date in extra_hols.items():
        curr_year = min_timestamp.year
        while curr_year <= max_timestamp.year:
            new_hol = pd.DataFrame({'ds':[datetime(curr_year,date[1],date[0])], 'holiday':[hol]})
            holidays_df = pd.concat([holidays_df, new_hol], axis=0)
            curr_year += 1

    return holidays_df


def ohe_holidays(holiday_subset:list, min_timestamp:datetime, max_timestamp:datetime) -> pd.DataFrame:
    """
    Creates One-Hot-Encoding (OHE) features for US holidays across a given time period.

    Args:
        holiday_subset (list): Set of holidays to OHE
        min_timestamp (datetime): Minimum timestamp of the time period.
        max_timestamp (datetime): Maximum timestamp of the time period.

    Returns:
        pd.DataFrame: Binary dummy variable DataFrame for each of the given holidays.
    """

    holidays_df = get_holidays(min_timestamp, max_timestamp)

    # Expand holidays df to hourly basis
    hours = pd.DataFrame({'hour': range(24)})
    hourly_holidays_df = holidays_df.merge(hours, how='cross')
    hourly_holidays_df['ds'] = hourly_holidays_df['ds'] + pd.to_timedelta(hourly_holidays_df['hour'], unit='h')

    # Create hourly index across our date range
    hourly_index = pd.date_range(start=min_timestamp, end=max_timestamp, freq='1h')  
    ts = pd.DataFrame(index=hourly_index)
    logger.debug(f"Adding holidays in period from {ts.index.min()} to {ts.index.max()}")

    # Create One Hot Encodings for each holiday
    holiday_ohe = pd.get_dummies(hourly_holidays_df.set_index('ds')['holiday'], dtype=int)
    holiday_hourly = holiday_ohe.reindex(ts.index, fill_value=0)
    holiday_hourly_cut = holiday_hourly[holiday_subset].fillna(0) # subset

    return holiday_hourly_cut


def lag_features(df:pd.DataFrame, lag_dict:dict) -> pd.DataFrame:
    """
    Creates a set of lagged features from given time-series columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing time-series data.
        lag_dict (dict): Dictionary where keys are column names to create lags for, and values are lists of lag periods.

    Returns:
        pd.DataFrame: Input DataFrame + lagged features
    """

    df_lags = df.copy()

    lag_cols = lag_dict.keys()

    for col in lag_cols:
        if col in df_lags.columns:
            for lag in lag_dict[col]:
                df_lags[f'{col}_lag_{lag}'] = df_lags[col].shift(lag)

    return df_lags


def rolling_window_features(df:pd.DataFrame, rw_dict:dict, agg_func:Literal["mean","sum"]) -> pd.DataFrame:
    """
    Creates rolling window features from given time-series columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing time-series data.
        rw_dict (dict): Dictionary where keys are column names to create rolling windows for, and values are lists of window sizes.
        agg_func (str): Aggregation function to apply over the rolling window (e.g., 'mean', 'sum').

    Returns:
        pd.DataFrame: Input DataFrame + rolling window features
    """
    df_rw = df.copy()

    rw_cols = rw_dict.keys()

    for col in rw_cols:
        for window in rw_dict[col]:
            df_rw[f"{col}_rw_{window}_{agg_func}"] = (
                df_rw[col]
                .shift(1)
                .rolling(window)
                .agg(agg_func)
            )

    return df_rw


def flatten_nested_dict(nest_dict:dict) -> list:
    """
    Flattens a nested dictionary into a list of integer values.

    Args:
        nest_dict (dict): Input nested dictionary.

    Returns:
        list: Flattened list of integer values.
    """

    stack = [nest_dict]
    flattened_list = []

    while stack:
        current = stack.pop()

        if isinstance(current, dict):
            stack.extend(current.values())

        elif isinstance(current, list):
            stack.extend(current)

        elif isinstance(current, int):
            flattened_list.append(current)

    return flattened_list

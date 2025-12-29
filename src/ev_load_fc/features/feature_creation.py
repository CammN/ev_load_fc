import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calender
import logging
logger = logging.getLogger(__name__)



def aggregate_features(df:pd.DataFrame, out_name:str, substr1:str, substr2:str='') -> pd.DataFrame:

    df_agg = df.copy()

    fname_cols = [col for col in df_agg.columns if (substr1 in col) and (substr2 in col)]
    
    df_agg[out_name] = 0
    for col in fname_cols:
        df_agg[out_name] += df_agg[col]

    return df_agg


def date_features(df:pd.DataFrame) -> pd.DataFrame:

    df_dates = df.copy()

    df_dates['month_sin'] = np.sin(2 * np.pi * df_dates.index.month/12) 
    df_dates['month_cos'] = np.cos(2 * np.pi * df_dates.index.month/12) 

    df_dates['weekday_sin'] = np.sin(2 * np.pi * df_dates.index.weekday/7) 
    df_dates['weekday_cos'] = np.cos(2 * np.pi * df_dates.index.weekday/7) 

    df_dates['hour_sin'] = np.sin(2 * np.pi * df_dates.index.hour/24) 
    df_dates['hour_cos'] = np.cos(2 * np.pi * df_dates.index.hour/24)

    return df_dates


def get_holidays(holiday_subset, min_timestamp:datetime, max_timestamp:datetime) -> pd.DataFrame:

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

    df_lags = df.copy()

    lag_cols = lag_dict.keys()

    for col in lag_cols:
        for lag in lag_dict[col]:
            df_lags[f'{col}_lag_{lag}'] = df_lags[col].shift(lag)

    return df_lags


def rolling_window_features(df:pd.DataFrame, rw_dict:dict, agg_func:str) -> pd.DataFrame:

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


def flatten_nested_dict(nest_dict):
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

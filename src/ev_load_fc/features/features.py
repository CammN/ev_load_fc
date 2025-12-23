import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression


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


def f_score_df(X, y):

    f_scores = pd.DataFrame()

    for col in X.columns:
        f = f_regression()
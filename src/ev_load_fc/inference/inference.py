import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from prophet import Prophet
from ev_load_fc.config import CFG
from ev_load_fc.features.feature_creation import (
    lag_features, 
    rolling_window_features,
    flatten_nested_dict,
)
from typing import Union, List

HOLIDAYS = list(CFG["features"]["feature_engineering"]["holidays"])

def sarimax_one_step(
        fitted_model:SARIMAXResults, 
        y_test:pd.Series, 
        start:pd.Timestamp|int, 
        end:pd.Timestamp|int,
        score_func:Union[root_mean_squared_error, mean_absolute_error],
    ) -> np.float64:

    if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
        step_range = pd.date_range(start, end)
    elif isinstance(start, int) and isinstance(end, int):
        step_range = range(start+1, end+1)
    else:
        raise ValueError("One or both of start and end have incompatible types, they must both be int or both be datetime.")
    
    scores = []
    for i, step in enumerate(step_range):
        forecast = fitted_model.forecast(steps=1)
        score = score_func(y_test[i],forecast)
        scores.append(score)
        fitted_model = fitted_model.append([y_test[i]], refit=False)

    return np.mean(scores)


def get_feature_set(fitted_model) -> List[str]:
    """Extracts input features used to fit model, logic depends on model API
        - XGBRegressor
        - RandomForestRegressor
        - CatBoostRegression
        - LGBMRegressor
        - Prophet
        - AdaBoostRegressor

    Args:
        fitted_model (_type_): Fitted model object to extract feature set from
    """
    if hasattr(fitted_model, 'feature_names_in_'):
        feature_set = list(fitted_model.feature_names_in_)
    elif hasattr(fitted_model, 'feature_names_'):
        feature_set = list(fitted_model.feature_names_)    
    elif isinstance(fitted_model, Prophet):
        feature_set = list(fitted_model.extra_regressors.keys())
    else:
        raise ValueError("Model type not recognized, unable to extract feature set.")
    
    return feature_set

def recursive_forecast(
        fitted_model,
        raw_hourly:pd.DataFrame,
        X:pd.DataFrame, 
        horizon:int, 
        forecast_start:pd.Timestamp, 
    ) -> pd.Series:
    """Performs a recursive multi-step forecast using a fitted model, by iteratively predicting one step ahead and then updating the input features with the new prediction before predicting the next step.

    Args:
        fitted_model (_type_): A fitted model object with a predict method and accessible feature set
        raw_hourly (pd.DataFrame): Contains historical pre-feature target and exogenous data - target features will be computed from this on a rolling basis at each step in forecast
        X (pd.DataFrame): Contains precomputed features of target and exogenous data i.e. lags, rolling window, time features
            - Only features created from non-target data (i.e. weather, traffic) will be used, target based features (e.g. energy_lag_24) will be computed from raw_hourly on a rolling basis
        horizon (int): Number of steps to forecast, in same frequency as raw_hourly and X (e.g. if hourly data, horizon of 24 means forecast 24 hours ahead)
        forecast_start (pd.Timestamp): Timestamp of first point in forecast horizon, must be after end of historical data in raw_hourly

    Raises:
        ValueError: If :
            - Forecast_start is before end of raw_hourly data
            - Required timestamps are missing from raw_hourly
            - Model type is not recognized in get_feature_set function

    Returns:
        pd.Series: Forecasted values for target variable over the forecast horizon, indexed by timestamp
    """

    feature_set  = get_feature_set(fitted_model)
    feature_set  = [str(feat) for feat in feature_set if str(feat) in X.columns and str(feat) != 'energy'] # restrict to features we have precomputed in X, and exclude energy column itself as this is what we will be updating in our rolling data
    feature_set += HOLIDAYS
    energy_feature_set = [feat for feat in feature_set if "energy" in feat]
    raw_energy_cols = [feat for feat in raw_hourly.columns if "energy" in feat]

    ### Dictionary containing lags to use when creating features from the energy column
    tfd = CFG["features"]["feature_engineering"]["time_feature_dict"]
    lag_dict = {}
    rw_sum_dict = {}
    rw_mean_dict = {}
    for col in raw_energy_cols:    
        if "energy" in tfd["lags"].keys():
            lag_dict[col] = tfd["lags"]["energy"]    
        if "energy" in tfd["rolling_sums"].keys():
            rw_sum_dict[col] = tfd["rolling_sums"]["energy"]    
        if "energy" in tfd["rolling_means"].keys():
            rw_mean_dict[col] = tfd["rolling_means"]["energy"]
            
    # Find the max possible lag so we can restrict the data we work with
    all_lags = flatten_nested_dict(tfd)
    max_lag  = max(all_lags)

    # Create date range covering period from when our earliest lags exist to the end of our forecast horizon - used to filter X_raw
    lag_start = forecast_start - pd.Timedelta(hours=max_lag+1)
    total_hours = max_lag + horizon + 1 
    dates = pd.date_range(lag_start, periods=total_hours, freq='h')

    missing_dates = [d for d in dates if d not in raw_hourly.index]
    if missing_dates:
        raise ValueError(f"raw_hourly is missing {len(missing_dates)} required timestamps, earliest: {min(missing_dates)}")
    
    # df storing our pre-feature engineering data, updated each step in forecast with predicted value for the associated point in time
    rolling_data = raw_hourly.loc[dates].copy()
    rolling_data.loc[rolling_data.index >= forecast_start, 'energy'] = np.nan
    # df to store updated energy features and pre-existing exogenous features
    X_rolling = X[feature_set].loc[dates].copy()

    forecast = []
    current_time = forecast_start
    for step in range(horizon):

        # 1 step prediction
        X_step = X_rolling.iloc[[step]]
        X_step = X_step.reindex(columns=feature_set, fill_value=0)
        step_y = fitted_model.predict(X_step)[0]
        forecast.append(step_y)

        # Overwrite rolling data with prediction
        rolling_data.at[current_time, 'energy'] = step_y
        current_time += pd.Timedelta(hours=1)

        # Recompute lag/rolling window features with updated prediction
        rolling_data = lag_features(rolling_data, lag_dict)
        rolling_data = rolling_window_features(rolling_data, rw_sum_dict, 'sum')
        rolling_data = rolling_window_features(rolling_data, rw_mean_dict, 'mean')

        # Update our rolling X_test df with recomputed features
        shared_energy_feats = [col for col in energy_feature_set if col in X_rolling.columns]
        X_rolling[shared_energy_feats] = rolling_data.loc[X_rolling.index, shared_energy_feats].copy()

        # Reindex to preserve original index
        rolling_data = rolling_data.reindex(dates)

    fc_series = pd.Series(forecast, index=pd.date_range(forecast_start, periods=horizon, freq='h'))

    # Add back trend if model was trained on detrended data
    if CFG["features"]["detrend"]:
        slope = CFG["inference"]["slope"]
        intercept = CFG["inference"]["intercept"]
        t_future = np.arange(len(raw_hourly), len(raw_hourly) + horizon)
        trend_future = slope * t_future + intercept
        fc_series += trend_future

    fc_df = pd.DataFrame({'timestamp': fc_series.index, 'yhat': fc_series.values})
    fc_df['y'] = raw_hourly.loc[fc_df['timestamp'], 'energy'].values

    return fc_df


import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from ev_load_fc.config import CFG
from ev_load_fc.features.feature_creation import (
    lag_features, 
    rolling_window_features,
    flatten_nested_dict,
)

def sarimax_one_step(
        fitted_model:SARIMAXResults, 
        y_test:pd.Series, 
        start, 
        end
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
        rsme = root_mean_squared_error(y_test[i],forecast)
        scores.append(rsme)
        fitted_model = fitted_model.append([y_test[i]], refit=False)

    return np.mean(scores)


def recursive_forecast(
        fitted_model, 
        pre_features:pd.DataFrame,
        X_test:pd.DataFrame, 
        horizon:int, 
        forecast_start:pd.Timestamp, 
    ) -> pd.Series:

    column_set = X_test.columns
    energy_feats = [col for col in column_set if 'energy' in col]

    ### Dictionary containing lags to use when creating features from the energy column
    tfd = CFG["features"]["feature_engineering"]["time_feature_dict"]
    lag_dict = {}
    rw_sum_dict = {}
    rw_mean_dict = {}
    for col in energy_feats:    
        if "energy" in tfd["lags"].keys():
            lag_dict[col] = tfd["rolling_sums"]["energy"]    
        if "energy" in tfd["rolling_sums"].keys():
            rw_sum_dict[col] = tfd["lags"]["energy"]    
        if "energy" in tfd["rolling_means"].keys():
            rw_mean_dict[col] = tfd["rolling_means"]["energy"]
            
    # Find the max possible lag so we can restrict the data we work with
    all_lags = flatten_nested_dict(tfd)
    max_lag  = max(all_lags)

    # Create date range covering period from when our earliest lags exist to the end of our forecast horizon 
    start_time = forecast_start - pd.Timedelta(hours=max_lag+1)
    total_hours = max_lag + horizon
    dates = pd.date_range(start_time, periods=total_hours, freq='h')
    
    # df storing our pre-feature engineering data, updated each step in forecast with predicted value for the associated point in time
    rolling_data = pre_features.loc[dates].copy()
    # df to store updated energy features and pre-existing exogenous features
    X_rolling = X_test.loc[X_test.index >= forecast_start].copy()

    forecast = []
    current_time = forecast_start
    for step in range(horizon):

        # 1 step prediction
        step_y = fitted_model.predict(X_rolling.iloc[[step]])[0]
        forecast.append(step_y)

        # Overwrite rolling data with prediction
        rolling_data.at[current_time, 'energy'] = step_y
        current_time += pd.Timedelta(hours=1)

        # Recompute lag/rolling window features with updated prediction
        rolling_data = lag_features(rolling_data, lag_dict)
        rolling_data = rolling_window_features(rolling_data, rw_sum_dict, 'sum')
        rolling_data = rolling_window_features(rolling_data, rw_mean_dict, 'mean')

        # Update our rolling X_test df with recomputed features
        X_rolling[energy_feats] = rolling_data[energy_feats].copy()

    forecast_series = pd.Series(forecast, index=pd.date_range(forecast_start, periods=horizon, freq='h'))

    return forecast_series

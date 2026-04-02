import math

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def prophet_df_format(ts: pd.Series) -> pd.DataFrame:
    """Reformat a pd.Series (timestamp index, energy values) into Prophet's {ds, y} format.

    Args:
        ts (pd.Series): Time series with a DatetimeIndex and a named value column.

    Returns:
        pd.DataFrame: Two-column DataFrame with columns ``ds`` and ``y``.
    """
    df = ts.copy().reset_index()
    df.columns = ["ds", "y"]
    return df[["ds", "y"]]


def cv_score_prophet_model(model: Prophet, y: pd.Series, n_splits: int) -> dict:
    """Cross-validate a Prophet model and return mean RMSE and MAE across splits.

    Args:
        model (Prophet): A Prophet model instance.
        y (pd.Series): Input time series data.
        n_splits (int): Number of CV splits.

    Returns:
        dict: ``{"rmse": float, "mae": float}`` mean scores across CV splits.
    """
    y_proph = prophet_df_format(y) if isinstance(y, pd.Series) else y[["ds", "y"]].copy()

    model.fit(y_proph.dropna())
    total_days = (y_proph["ds"].max() - y_proph["ds"].min()).days
    window_length = math.floor((total_days - 366) / n_splits)
    cv = cross_validation(
        model=model,
        initial="366 days",
        horizon=f"{window_length} days",
        period=f"{window_length} days",
        disable_tqdm=True,
    )

    split_rmses = {}
    split_maes = {}
    for cutoff in cv["cutoff"].unique():
        split = cv[cv["cutoff"] == cutoff]
        min_date, max_date = split["ds"].min(), split["ds"].max()
        actual = y_proph[y_proph["ds"].between(min_date, max_date, inclusive="both")]
        split_rmses[cutoff] = root_mean_squared_error(actual["y"], split["yhat"])
        split_maes[cutoff] = mean_absolute_error(actual["y"], split["yhat"])

    return {
        "rmse": np.mean(list(split_rmses.values())),
        "mae": np.mean(list(split_maes.values())),
    }

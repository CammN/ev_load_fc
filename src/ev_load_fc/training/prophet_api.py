import math

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def get_prophet_regressor_cols(columns: list, cfg: dict) -> list:
    """Return column names that correspond to external regressors for Prophet.

    Filters to weather, temperature, and traffic columns using the substring
    lists defined in cfg["features"]["feature_engineering"]. Energy lags and
    rolling energy features are intentionally excluded.

    Args:
        columns: Full list of column names to filter.
        cfg: Project config dict (ev_load_fc.config.CFG).

    Returns:
        List of column names that match any of the external-regressor substrings.
    """
    fe = cfg["features"]["feature_engineering"]
    substrs = (
        fe["weather_col_substrs"]
        + fe["temperature_col_substrs"]
        + fe["traffic_col_substrs"]
    )
    return [c for c in columns if any(s in c for s in substrs)]


def prophet_df_format(ts: pd.Series, regressors_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reformat a pd.Series (timestamp index, energy values) into Prophet's {ds, y} format.

    Optionally appends external regressor columns alongside ds and y. Both ts
    and regressors_df must share the same DatetimeIndex (positional alignment
    is used after reset_index).

    Args:
        ts: Time series with a DatetimeIndex and a named value column.
        regressors_df: Optional DataFrame of external regressor columns, aligned
            to ts on the same DatetimeIndex.

    Returns:
        DataFrame with columns ``ds``, ``y``, and (optionally) regressor columns.
    """
    df = ts.copy().reset_index()
    df.columns = ["ds", "y"]
    if regressors_df is not None:
        reg = regressors_df.copy().reset_index(drop=True)
        df = pd.concat([df, reg], axis=1)
    return df[list(df.columns)]


def cv_score_prophet_model(
    model: Prophet,
    y: pd.Series,
    n_splits: int,
    regressors_df: pd.DataFrame | None = None,
    model_factory=None,
) -> dict:
    """Cross-validate a Prophet model and return mean RMSE and MAE across splits.

    Two CV paths:
    - Univariate (regressors_df is None): uses prophet.diagnostics.cross_validation.
    - With regressors: manual fold loop using model_factory to create a fresh
      model per fold (required because cross_validation cannot inject regressor
      values for holdout periods).

    Args:
        model: A Prophet model instance (used for univariate path).
        y: Input time series data.
        n_splits: Number of CV splits.
        regressors_df: Optional DataFrame of external regressor columns aligned to y.
        model_factory: Zero-arg callable returning a fresh Prophet model with
            add_regressor() already called. Required when regressors_df is provided.

    Returns:
        dict: ``{"rmse": float, "mae": float}`` mean scores across CV splits.

    Raises:
        ValueError: If regressors_df is provided but model_factory is None.
    """
    if regressors_df is not None and not regressors_df.empty:
        if model_factory is None:
            raise ValueError(
                "model_factory must be provided when regressors_df is not None. "
                "Pass a zero-arg callable that returns a fresh Prophet model with "
                "add_regressor() already called."
            )

        full_df = prophet_df_format(y, regressors_df)
        regressor_cols = [c for c in full_df.columns if c not in ("ds", "y")]

        total_days = (full_df["ds"].max() - full_df["ds"].min()).days
        window_length = math.floor((total_days - 366) / n_splits)
        initial_cutoff = full_df["ds"].min() + pd.Timedelta(days=366)

        rmses, maes = [], []
        for i in range(n_splits):
            cutoff = initial_cutoff + pd.Timedelta(days=i * window_length)
            fold_end = cutoff + pd.Timedelta(days=window_length)

            train_fold = full_df[full_df["ds"] <= cutoff].dropna()
            val_fold = full_df[(full_df["ds"] > cutoff) & (full_df["ds"] <= fold_end)]

            if val_fold.empty:
                continue

            fold_model = model_factory()
            fold_model.fit(train_fold)

            future = val_fold[["ds"] + regressor_cols]
            preds = fold_model.predict(future)

            rmses.append(root_mean_squared_error(val_fold["y"], preds["yhat"]))
            maes.append(mean_absolute_error(val_fold["y"], preds["yhat"]))

        return {
            "rmse": float(np.mean(rmses)),
            "mae": float(np.mean(maes)),
        }

    # --- Univariate path ---
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

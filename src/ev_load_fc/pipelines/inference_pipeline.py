import json
import pathlib
import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from ev_load_fc.data.loading import col_standardisation
from ev_load_fc.training.mlflow_api import init_mlflow, get_best_model
from prophet import Prophet
from ev_load_fc.inference.inference import HOLIDAYS, recursive_forecast, prophet_forecast
from ev_load_fc.features.feature_creation import flatten_nested_dict
from ev_load_fc.config import CFG

import logging
logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return s.lower().replace(" ", "_").replace("'", "").replace("-", "_")


@dataclass
class InferencePipelineConfig:
    # Paths
    raw_hourly_path: pathlib.Path
    feature_store: pathlib.Path
    predictions_dir: pathlib.Path
    X_set: str
    # MLflow model selection
    experiment_name: str
    model_family: str           # Optional: e.g. "XGBoost". Empty string = no filter.
    feature_version: str        # Optional: e.g. "E_f_30_rfe_20". Empty string = no filter.
    metric: str                 # Required: "rmse" or "mae" — ranks by metrics.best_{metric}
    # Forecast settings
    horizon: int
    inference_start: pd.Timestamp
    # Confidence interval settings
    confidence_intervals: list  # e.g. [0.80, 0.95]


class InferencePipeline:

    def __init__(self, config: InferencePipelineConfig):
        self.cfg = config
        self.model = None
        self.best_run = None
        self.raw_hourly = None
        self.X = None

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def _build_filter_string(self) -> str:
        """Build MLflow filter string from config tag filters.

        Always filters to parent-level runs (which have the model artifact).
        Optionally adds model_family and feature_set_version tag filters.
        """
        parts = ['tags.level = "parent"']

        if self.cfg.model_family:
            parts.append(f'tags.model_family = "{self.cfg.model_family}"')

        if self.cfg.feature_version:
            parts.append(f'tags.feature_set_version = "{self.cfg.feature_version}"')

        return " AND ".join(parts)

    def _load_model(self):
        """Retrieve the best model from MLflow based on configured filters and metric."""
        filter_string = self._build_filter_string()
        # Parent runs log metrics as "best_rmse" / "best_mae"
        mlflow_metric = f"best_{self.cfg.metric}"

        logger.info(
            f"Searching experiment '{self.cfg.experiment_name}' with "
            f"filter='{filter_string}', ranking by metrics.{mlflow_metric} ASC"
        )

        self.model, self.best_run = get_best_model(
            experiment_name=self.cfg.experiment_name,
            metric=mlflow_metric,
            ascending=True,     # lower RMSE/MAE is better
            filter_string=filter_string,
        )
        logger.info(f"Loaded model from run: {self.best_run['run_id']}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load raw_hourly and the feature matrix X for the configured feature version."""

        # Resolve the feature version to use for X
        feature_ver = self.cfg.feature_version or self.best_run.get("tags.feature_set_version", "")
        if not feature_ver:
            raise ValueError(
                "feature_version is not set in config and could not be inferred from the best run tags."
            )
        self._resolved_feature_version = feature_ver

        # Load raw hourly data (pre-feature-engineered time series)
        logger.info(f"Loading raw_hourly from {self.cfg.raw_hourly_path}")
        self.raw_hourly = pd.read_csv(
            self.cfg.raw_hourly_path,
            parse_dates=["timestamp"],
            index_col="timestamp",
        )

        # Load precomputed feature matrix X
        x_path = self.cfg.feature_store / f"{self.cfg.X_set}.csv"
        logger.info(f"Loading X from {x_path}")
        self.X = pd.read_csv(x_path, parse_dates=["timestamp"], index_col="timestamp")

        # recursive_forecast needs lag history going back max_lag hours before inference_start.
        # If X_set is a test-only file (starts at the train/test split), prepend the
        # corresponding train file so that lag timestamps are available.
        if self.cfg.X_set.startswith("test_"):
            train_x_name = self.cfg.X_set.replace("test_", "train_", 1)
            train_x_path = self.cfg.feature_store / f"{train_x_name}.csv"
            if train_x_path.exists():
                logger.info(f"Prepending train X from {train_x_path} to cover lag timestamps")
                train_X = pd.read_csv(train_x_path, parse_dates=["timestamp"], index_col="timestamp")
                self.X = pd.concat([train_X, self.X]).sort_index()

        logger.info(
            f"raw_hourly shape: {self.raw_hourly.shape}, "
            f"X shape: {self.X.shape}"
        )

    # ------------------------------------------------------------------
    # Confidence interval estimation
    # ------------------------------------------------------------------

    def _get_model_features(self) -> list:
        """Return the feature names the model was trained on."""
        if hasattr(self.model, "feature_names_in_"):        # sklearn / RF
            return list(self.model.feature_names_in_)
        if hasattr(self.model, "feature_name_"):            # LightGBM sklearn API
            return list(self.model.feature_name_)
        if hasattr(self.model, "feature_name"):             # LightGBM booster method
            return list(self.model.feature_name())
        if hasattr(self.model, "get_booster"):              # XGBoost
            return list(self.model.get_booster().feature_names)
        if hasattr(self.model, "feature_names_"):           # CatBoost
            return list(self.model.feature_names_)
        logger.warning("Could not determine training features from model; using all X columns.")
        return list(self.X.columns)

    def _rf_ci(self, X_forecast: pd.DataFrame) -> pd.DataFrame:
        """Prediction intervals for Random Forest via individual tree variance.

        Each tree in the forest is an independent bootstrap estimator, so the
        spread of tree predictions is a valid measure of epistemic uncertainty.
        Intervals: yhat ± z * std(tree_predictions).
        """
        tree_preds = np.array([est.predict(X_forecast) for est in self.model.estimators_])
        std = np.std(tree_preds, axis=0)

        ci_df = pd.DataFrame(index=X_forecast.index)
        for level in self.cfg.confidence_intervals:
            z = stats.norm.ppf((1 + level) / 2)
            label = int(level * 100)
            ci_df[f"yhat_lower_{label}"] = -z * std
            ci_df[f"yhat_upper_{label}"] = z * std
        return ci_df

    def _build_conformal_ci_df(self, residuals: np.ndarray, index) -> pd.DataFrame:
        """Build a CI DataFrame from absolute residuals using empirical quantiles.

        Uses finite-sample correction (1 + 1/n) so marginal coverage ≥ nominal level.
        """
        n = len(residuals)
        ci_df = pd.DataFrame(index=index)
        for level in self.cfg.confidence_intervals:
            q_level = min((1 + 1 / n) * level, 1.0)
            q = float(np.quantile(residuals, q_level, method="higher"))
            label = int(level * 100)
            ci_df[f"yhat_lower_{label}"] = -q
            ci_df[f"yhat_upper_{label}"] = q
            logger.info(f"Conformal {label}% CI half-width: {q:.3f} kWh (n_cal={n})")
        return ci_df

    def _conformal_ci(self, X_forecast: pd.DataFrame) -> pd.DataFrame:
        """Split conformal prediction intervals using training residuals.

        Loads the training feature set, computes absolute residuals with the
        current model, then uses empirical quantiles (with finite-sample
        correction) as symmetric CI half-widths. Provides a marginal coverage
        guarantee under exchangeability — the intervals reflect actual forecast
        error magnitude rather than model convergence noise.
        """
        target = CFG["features"]["target"]
        train_path = self.cfg.feature_store / f"train_{self._resolved_feature_version}.csv"
        if not train_path.exists():
            logger.warning(f"Training data not found at {train_path}; skipping CI.")
            return pd.DataFrame(index=X_forecast.index)

        train_df = pd.read_csv(train_path, parse_dates=["timestamp"], index_col="timestamp")
        training_features = self._get_model_features()

        missing = [f for f in training_features if f not in train_df.columns]
        if missing:
            logger.warning(f"{len(missing)} training features missing from train file; skipping CI.")
            return pd.DataFrame(index=X_forecast.index)

        X_cal = train_df[training_features]
        y_cal = train_df[target].values
        residuals = np.abs(y_cal - self.model.predict(X_cal))
        return self._build_conformal_ci_df(residuals, X_forecast.index)

    def _compute_ci(self, X_forecast: pd.DataFrame) -> pd.DataFrame:
        """Compute prediction interval bounds for each forecast step.

        Strategy by model type:
        - RandomForest   : variance of individual tree predictions (valid epistemic uncertainty)
        - LightGBM       : split conformal using training residuals (marginal coverage guarantee)
        - XGBoost        : split conformal using training residuals (marginal coverage guarantee)
        - CatBoost       : split conformal using training residuals (marginal coverage guarantee)

        Returns a DataFrame with columns yhat_lower_X and yhat_upper_X for each CI level.
        """
        training_features = self._get_model_features()

        norm_to_model = {_norm(f): f for f in training_features}
        col_remap = {
            col: norm_to_model[_norm(col)]
            for col in X_forecast.columns
            if col not in training_features and _norm(col) in norm_to_model
        }
        if col_remap:
            X_forecast = X_forecast.rename(columns=col_remap)

        X_forecast = X_forecast[training_features]

        model_type = type(self.model).__name__

        if "RandomForest" in model_type:
            logger.info("Computing CI via individual tree variance (Random Forest)")
            return self._rf_ci(X_forecast)
        elif "XGB" in model_type or "LGBM" in model_type or "LightGBM" in model_type or "CatBoost" in model_type:
            logger.info(f"Computing CI via split conformal (training residuals) for {model_type}")
            return self._conformal_ci(X_forecast)
        else:
            logger.warning(f"CI not supported for model type '{model_type}'. Skipping.")
            return pd.DataFrame(index=X_forecast.index)

    def _conformal_ci_prophet(self, fc_df: pd.DataFrame) -> pd.DataFrame:
        """Split conformal CI for Prophet using training residuals.

        Prophet has no feature matrix, so residuals are computed by re-predicting
        over the training period and comparing against the known target values.
        If the model was trained with external regressors, those values are pulled
        from self.X for the training timestamps.
        """
        target = CFG["features"]["target"]
        split_date = pd.Timestamp(CFG["data"]["preprocessing"]["split_date"])
        train_actuals = self.raw_hourly.loc[self.raw_hourly.index < split_date, target]

        if train_actuals.empty:
            logger.warning("No training actuals available for Prophet conformal CI; skipping.")
            return pd.DataFrame(index=fc_df.index)

        train_future = pd.DataFrame({"ds": train_actuals.index})

        # Include regressor columns if the model was trained with them
        regressor_cols = list(self.model.extra_regressors.keys())
        if regressor_cols:
            train_timestamps = train_actuals.index
            missing = [c for c in regressor_cols if c not in self.X.columns]
            if missing:
                logger.warning(
                    f"self.X is missing {len(missing)} regressor column(s) needed for "
                    f"conformal CI re-prediction: {missing}. Skipping CI."
                )
                return pd.DataFrame(index=fc_df.index)
            x_missing_ts = [t for t in train_timestamps if t not in self.X.index]
            if x_missing_ts:
                logger.warning(
                    f"self.X is missing {len(x_missing_ts)} training timestamps; "
                    f"conformal CI residuals may be incomplete."
                )
                return pd.DataFrame(index=fc_df.index)
            reg_vals = self.X.loc[train_timestamps, regressor_cols].reset_index(drop=True)
            train_future = pd.concat(
                [train_future.reset_index(drop=True), reg_vals], axis=1
            )

        train_preds = self.model.predict(train_future)
        residuals = np.abs(train_actuals.values - train_preds["yhat"].values)
        return self._build_conformal_ci_df(residuals, range(len(fc_df)))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save_predictions(self, fc_df: pd.DataFrame, run_id: str) -> pathlib.Path:
        """Save forecast DataFrame and metadata to a structured directory under datasets/05_predictions/.

        Directory name format:
            {YYYYMMDD_HHMMSS}__{ModelFamily}__{horizon}h__from_{start_YYYYMMDD}

        Each run directory contains:
            predictions.csv  — forecast DataFrame (timestamp, yhat, y, CI columns)
            metadata.json    — run configuration and summary metrics
        """
        created_at = datetime.datetime.now()
        model_family = self.best_run.get("tags.model_family", "Unknown")
        start_str = self.cfg.inference_start.strftime("%Y%m%d")
        dir_name = (
            f"{created_at.strftime('%Y%m%d_%H%M%S')}"
            f"__{model_family}"
            f"__{self.cfg.horizon}h"
            f"__from_{start_str}"
        )

        out_dir = self.cfg.predictions_dir / dir_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions CSV
        out_path = out_dir / "predictions.csv"
        fc_df.to_csv(out_path, index=False)

        # Compute summary metrics if actuals are available
        metrics = {}
        if "y" in fc_df.columns and fc_df["y"].notna().any():
            mask = fc_df["y"].notna()
            residuals = fc_df.loc[mask, "y"] - fc_df.loc[mask, "yhat"]
            metrics["rmse"] = float(np.sqrt((residuals ** 2).mean()))
            metrics["mae"] = float(residuals.abs().mean())

        # Save metadata JSON
        metadata = {
            "run_id": run_id,
            "experiment_name": self.cfg.experiment_name,
            "model_family": model_family,
            "feature_version": getattr(self, "_resolved_feature_version", self.cfg.feature_version),
            "horizon": self.cfg.horizon,
            "inference_start": self.cfg.inference_start.isoformat(),
            "ci_levels": self.cfg.confidence_intervals,
            "metrics": metrics,
            "created_at": created_at.isoformat(),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Predictions saved to {out_path}")
        return out_path

    def _validate_inference_start(self):
        """Validate that inference_start lies within raw_hourly and has sufficient lag history.

        Raises:
            ValueError: If inference_start is outside the data range, or if there is
                        insufficient historical data before it to compute all lag features.
        """
        data_min = self.raw_hourly.index.min()
        data_max = self.raw_hourly.index.max()

        if self.cfg.inference_start < data_min or self.cfg.inference_start > data_max:
            raise ValueError(
                f"inference_start {self.cfg.inference_start} is outside the available data range "
                f"[{data_min}, {data_max}]."
            )

        tfd = CFG["features"]["feature_engineering"]["time_feature_dict"]
        max_lag = max(flatten_nested_dict(tfd))
        required_start = self.cfg.inference_start - pd.Timedelta(hours=max_lag)

        if required_start < data_min:
            shortfall = int((data_min - required_start).total_seconds() // 3600)
            raise ValueError(
                f"inference_start {self.cfg.inference_start} does not have enough historical data "
                f"for lag features. Requires {max_lag} hours of lookback (back to {required_start}), "
                f"but raw_hourly only starts at {data_min} — {shortfall} hours short. "
                f"Set inference_start to {data_min + pd.Timedelta(hours=max_lag)} or later."
            )

        logger.info(
            f"inference_start {self.cfg.inference_start} validated — "
            f"{max_lag}h lookback available (max lag required: {max_lag}h)."
        )

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:

        init_mlflow()

        self._load_model()
        self._load_data()
        self._validate_inference_start()

        logger.info(
            f"Running recursive forecast: start={self.cfg.inference_start}, horizon={self.cfg.horizon}"
        )

        if isinstance(self.model, Prophet):
            logger.info("Prophet model detected — using single-shot prophet_forecast")
            fc_df = prophet_forecast(
                fitted_model=self.model,
                raw_hourly=self.raw_hourly,
                forecast_start=self.cfg.inference_start,
                horizon=self.cfg.horizon,
                X=self.X,
            )
            X_forecast = None
        else:
            fc_df = recursive_forecast(
                fitted_model=self.model,
                raw_hourly=self.raw_hourly,
                X=self.X,
                horizon=self.cfg.horizon,
                forecast_start=self.cfg.inference_start,
            )
            X_forecast = self.X.loc[fc_df["timestamp"]]
            X_forecast = col_standardisation(X_forecast)

        # Compute confidence intervals
        if X_forecast is not None:
            ci_df = self._compute_ci(X_forecast)
        else:
            ci_df = self._conformal_ci_prophet(fc_df)

        # Attach CI columns — ci_df stores signed offsets: lower=-z*std, upper=+z*std
        for col in ci_df.columns:
            fc_df[col] = fc_df["yhat"].values + ci_df[col].values

        run_id = self.best_run["run_id"]
        self._save_predictions(fc_df, run_id)

        return fc_df
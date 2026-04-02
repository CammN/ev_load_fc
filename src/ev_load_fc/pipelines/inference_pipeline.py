import json
import pathlib
import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from ev_load_fc.data.loading import col_standardisation
from ev_load_fc.training.mlflow_api import init_mlflow, get_best_model
from ev_load_fc.inference.inference import HOLIDAYS, recursive_forecast
from ev_load_fc.features.feature_creation import flatten_nested_dict
from ev_load_fc.config import CFG

import logging
logger = logging.getLogger(__name__)


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
    n_bootstrap: int            # Number of bootstrap samples for gradient boosting CI


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
        if hasattr(self.model, "feature_names_in_"):        # sklearn / RF / AdaBoost
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

    def _rf_std(self, X_forecast: pd.DataFrame) -> np.ndarray:
        """Std of individual tree predictions for RF / AdaBoost."""
        tree_preds = np.array([est.predict(X_forecast) for est in self.model.estimators_])
        return np.std(tree_preds, axis=0)

    def _xgb_std(self, X_forecast: pd.DataFrame) -> np.ndarray:
        """Std via MC-dropout for XGBoost dart booster, or tree-subset sampling for gbtree."""
        import xgboost as xgb
        booster = self.model.get_booster()
        dm = xgb.DMatrix(X_forecast)

        # Check booster type from model params
        booster_type = self.model.get_params().get("booster", "gbtree")

        if booster_type == "dart":
            # training=True activates random tree dropout, giving stochastic predictions
            preds = np.array([booster.predict(dm, training=True) for _ in range(self.cfg.n_bootstrap)])
        else:
            n_trees = booster.num_boosted_rounds()
            half = max(1, n_trees // 2)
            preds = np.array([
                booster.predict(dm, iteration_range=(0, np.random.randint(half, n_trees + 1)))
                for _ in range(self.cfg.n_bootstrap)
            ])

        return np.std(preds, axis=0)

    def _lgbm_std(self, X_forecast: pd.DataFrame) -> np.ndarray:
        """Std via random tree-subset sampling for LightGBM."""
        # n_estimators_ is the actual number of fitted trees
        n_trees = getattr(self.model, "n_estimators_", self.model.n_estimators)
        half = max(1, n_trees // 2)
        preds = np.array([
            self.model.predict(X_forecast, num_iteration=np.random.randint(half, n_trees + 1))
            for _ in range(self.cfg.n_bootstrap)
        ])
        return np.std(preds, axis=0)

    def _catboost_std(self, X_forecast: pd.DataFrame) -> np.ndarray:
        """Std via random tree-subset sampling for CatBoost."""
        n_trees = self.model.tree_count_
        half = max(1, n_trees // 2)
        preds = np.array([
            self.model.predict(X_forecast, ntree_end=np.random.randint(half, n_trees + 1))
            for _ in range(self.cfg.n_bootstrap)
        ])
        return np.std(preds, axis=0)

    def _compute_ci(self, X_forecast: pd.DataFrame) -> pd.DataFrame:
        """Compute CI bounds for each forecast step using ensemble variance.

        Dispatches to the appropriate method based on model type:
        - RF / AdaBoost : variance of individual tree predictions
        - XGBoost (dart): MC-dropout via predict(training=True)
        - XGBoost (other): random tree-subset sampling
        - LightGBM       : random tree-subset sampling
        - CatBoost       : random tree-subset sampling

        Returns a DataFrame with columns yhat_lower_X and yhat_upper_X for each CI level.
        """
        training_features = self._get_model_features()

        holiday_features = {feat for feat in training_features if
                            feat[:3] in [h[:3] for h in HOLIDAYS]}

        # Normalise X_forecast column names to match model training names.
        # col_standardisation lowercases everything, but models may have been trained
        # with title-case holiday column names (e.g. "Independence_Day").
        # Build a case-insensitive mapping: normalised_col → model_feature_name.
        def _norm(s: str) -> str:
            return s.lower().replace(" ", "_").replace("'", "").replace("-", "_")

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

        if hasattr(self.model, "estimators_"):
            # RandomForestRegressor, AdaBoostRegressor
            logger.info("Computing CI via individual estimator variance (RF/AdaBoost)")
            std = self._rf_std(X_forecast)
        elif "XGB" in model_type:
            logger.info(f"Computing CI via XGBoost ensemble sampling (n_bootstrap={self.cfg.n_bootstrap})")
            std = self._xgb_std(X_forecast)
        elif "LGBM" in model_type or "LightGBM" in model_type:
            logger.info(f"Computing CI via LightGBM sub-ensemble sampling (n_bootstrap={self.cfg.n_bootstrap})")
            std = self._lgbm_std(X_forecast)
        elif "CatBoost" in model_type:
            logger.info(f"Computing CI via CatBoost sub-ensemble sampling (n_bootstrap={self.cfg.n_bootstrap})")
            std = self._catboost_std(X_forecast)
        else:
            logger.warning(f"CI not supported for model type '{model_type}'. Skipping.")
            return pd.DataFrame(index=X_forecast.index)

        ci_df = pd.DataFrame(index=X_forecast.index)
        for level in self.cfg.confidence_intervals:
            z = stats.norm.ppf((1 + level) / 2)
            label = int(level * 100)
            ci_df[f"yhat_lower_{label}"] = -z * std
            ci_df[f"yhat_upper_{label}"] = z * std

        return ci_df

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

        fc_df = recursive_forecast(
            fitted_model=self.model,
            raw_hourly=self.raw_hourly,
            X=self.X,
            horizon=self.cfg.horizon,
            forecast_start=self.cfg.inference_start,
        )

        # Compute confidence intervals using the precomputed X features at forecast steps
        X_forecast = self.X.loc[fc_df["timestamp"]]
        X_forecast = col_standardisation(X_forecast)
        ci_df = self._compute_ci(X_forecast)

        # Attach CI columns — ci_df stores signed offsets: lower=-z*std, upper=+z*std
        for col in ci_df.columns:
            fc_df[col] = fc_df["yhat"].values + ci_df[col].values

        run_id = self.best_run["run_id"]
        self._save_predictions(fc_df, run_id)

        return fc_df
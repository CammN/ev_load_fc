import pathlib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats

import mlflow

from ev_load_fc.training.mlflow_api import init_mlflow, get_best_model
from ev_load_fc.inference.inference import recursive_forecast

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
        x_path = self.cfg.feature_store / f"{self.cfg.X_set}_{feature_ver}.csv"
        logger.info(f"Loading X from {x_path}")
        self.X = pd.read_csv(x_path, parse_dates=["timestamp"], index_col="timestamp")

        logger.info(
            f"raw_hourly shape: {self.raw_hourly.shape}, "
            f"X shape: {self.X.shape}"
        )

    # ------------------------------------------------------------------
    # Confidence interval estimation
    # ------------------------------------------------------------------

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
        """Save forecast DataFrame to datasets/05_predictions/<run_id>_<feature_version>/predictions.csv."""
        out_dir = self.cfg.predictions_dir / f"{run_id}_{self._resolved_feature_version}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.csv"
        fc_df.to_csv(out_path, index=False)
        logger.info(f"Predictions saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:

        init_mlflow()

        self._load_model()
        self._load_data()

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
        ci_df = self._compute_ci(X_forecast)

        # Attach CI columns (offset from yhat)
        for col in ci_df.columns:
            if "lower" in col:
                fc_df[col] = fc_df["yhat"].values + ci_df[col].values
            else:
                fc_df[col] = fc_df["yhat"].values + ci_df[col].values

        run_id = self.best_run["run_id"]
        self._save_predictions(fc_df, run_id)

        return fc_df
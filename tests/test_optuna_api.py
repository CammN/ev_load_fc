"""Unit tests for ev_load_fc.training.optuna_api (Stage 2).

Verifies that objective() correctly:
- Passes external regressors to Prophet CV and final fit
- Falls back to univariate when no regressor columns are in X
- Logs a fitted Prophet model artifact with extra_regressors populated
- Handles edge cases (all-zero regressors, no regressors in X)
"""
import math
import tempfile
import pathlib
import numpy as np
import pandas as pd
import pytest
import mlflow
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from ev_load_fc.training.optuna_api import objective


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_train_df(n_days: int, reg_cols: list | None = None, seed: int = 0) -> pd.DataFrame:
    """Synthetic hourly training DataFrame with energy target + optional regressors."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days * 24, freq="h")
    t = np.arange(len(idx))
    energy = (
        50
        + 10 * np.sin(2 * np.pi * t / 24)
        + 5 * np.sin(2 * np.pi * t / (24 * 7))
        + rng.normal(0, 2, len(idx))
    )
    df = pd.DataFrame({"energy": energy}, index=idx)
    if reg_cols:
        for col in reg_cols:
            df[col] = rng.uniform(0, 1, len(idx))
    # Add non-regressor columns (time features) that should be ignored by Prophet
    df["hour_sin"] = np.sin(2 * np.pi * t / 24)
    df["energy_lag_24"] = df["energy"].shift(24)
    return df.dropna()


_MINIMAL_PROPHET_SEARCH_SPACE = {
    "seasonality_mode": ["additive"],
    "n_changepoints": [10, 10],          # int range → always 10
    "changepoint_prior_scale": [0.05, 0.05],  # float → always 0.05
}


def _run_one_trial(train_df, target="energy", reg_cols_present=True):
    """Run a single Optuna trial for Prophet and return (score, run_id)."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tracking_uri = f"sqlite:///{tmpdir}/mlflow_test.db"
        mlflow.set_tracking_uri(tracking_uri)
        exp_id = mlflow.create_experiment("test_optuna_prophet")

        with mlflow.start_run(experiment_id=exp_id, run_name="parent") as parent_run:
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: objective(
                    trial=trial,
                    train=train_df,
                    target=target,
                    model_name="Prophet",
                    search_space=_MINIMAL_PROPHET_SEARCH_SPACE,
                    n_splits=1,
                    metric="rmse",
                    experiment_id=exp_id,
                    parent_run_name="parent",
                    parent_run_id=parent_run.info.run_id,
                    feature_set_version="test_v1",
                    seed=None,
                ),
                n_trials=1,
            )
            # Return the child run id (the only child)
            runs = mlflow.search_runs(
                experiment_ids=[exp_id],
                filter_string='tags.level = "child"',
            )
            child_run_id = runs.iloc[0]["run_id"]
        return study.best_value, child_run_id, tracking_uri


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestObjectiveProphetWithRegressors:

    @pytest.fixture(scope="class")
    def train_with_regressors(self):
        return _make_train_df(500, reg_cols=["fog_dur_rw_1_sum", "temp_rw_1_mean"])

    def test_returns_finite_float(self, train_with_regressors):
        score, _, _ = _run_one_trial(train_with_regressors)
        assert np.isfinite(score) and score > 0

    def test_child_run_has_rmse_metric(self, train_with_regressors):
        _, child_id, uri = _run_one_trial(train_with_regressors)
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()
        metric = client.get_metric_history(child_id, "rmse")
        assert len(metric) == 1
        assert np.isfinite(metric[0].value)

    def test_logged_model_has_extra_regressors(self, train_with_regressors):
        """Final refit model artifact must have extra_regressors matching the reg cols."""
        _, child_id, uri = _run_one_trial(train_with_regressors)
        mlflow.set_tracking_uri(uri)
        local_path = mlflow.artifacts.download_artifacts(f"runs:/{child_id}/model")
        model = mlflow.prophet.load_model(pathlib.Path(local_path).as_uri())
        reg_keys = set(model.extra_regressors.keys())
        assert "fog_dur_rw_1_sum" in reg_keys
        assert "temp_rw_1_mean" in reg_keys

    def test_energy_lags_not_in_extra_regressors(self, train_with_regressors):
        """energy_lag_24 must never end up as a Prophet regressor."""
        _, child_id, uri = _run_one_trial(train_with_regressors)
        mlflow.set_tracking_uri(uri)
        local_path = mlflow.artifacts.download_artifacts(f"runs:/{child_id}/model")
        model = mlflow.prophet.load_model(pathlib.Path(local_path).as_uri())
        assert "energy_lag_24" not in model.extra_regressors

    def test_hour_sin_not_in_extra_regressors(self, train_with_regressors):
        """Time features must not end up as Prophet regressors."""
        _, child_id, uri = _run_one_trial(train_with_regressors)
        mlflow.set_tracking_uri(uri)
        local_path = mlflow.artifacts.download_artifacts(f"runs:/{child_id}/model")
        model = mlflow.prophet.load_model(pathlib.Path(local_path).as_uri())
        assert "hour_sin" not in model.extra_regressors

    def test_zero_regressor_no_crash(self):
        """All-zero regressor column must not cause Prophet to crash."""
        df = _make_train_df(500)
        df["fog_dur"] = 0.0
        score, _, _ = _run_one_trial(df)
        assert np.isfinite(score)


class TestObjectiveProphetWithoutRegressors:

    @pytest.fixture(scope="class")
    def train_no_regressors(self):
        return _make_train_df(500, reg_cols=None)

    def test_returns_finite_float_no_regressors(self, train_no_regressors):
        """Falls back to univariate Prophet when X has no weather/temp/traffic cols."""
        score, _, _ = _run_one_trial(train_no_regressors)
        assert np.isfinite(score) and score > 0

    def test_logged_model_has_no_extra_regressors(self, train_no_regressors):
        _, child_id, uri = _run_one_trial(train_no_regressors)
        mlflow.set_tracking_uri(uri)
        local_path = mlflow.artifacts.download_artifacts(f"runs:/{child_id}/model")
        model = mlflow.prophet.load_model(pathlib.Path(local_path).as_uri())
        assert len(model.extra_regressors) == 0

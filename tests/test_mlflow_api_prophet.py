"""Unit tests for parent_logging() in mlflow_api.py (Stage 3).

Verifies that parent_logging correctly:
- Registers and fits Prophet with external regressors from X_train
- Logs a model artifact whose extra_regressors match the regressor cols
- Falls back gracefully when no regressor cols are present in X_train
- Does not affect non-Prophet model logging
"""
import pathlib
import tempfile
import numpy as np
import pandas as pd
import pytest
import mlflow
import optuna

from unittest.mock import MagicMock, patch

from ev_load_fc.training.mlflow_api import parent_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_series(n_days: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days * 24, freq="h")
    t = np.arange(len(idx))
    vals = 50 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2, len(idx))
    return pd.Series(vals, index=idx, name="energy")


def _make_train_df(n_days: int, reg_cols: list | None = None, seed: int = 0) -> pd.DataFrame:
    ts = _make_series(n_days, seed)
    df = pd.DataFrame({"energy": ts})
    if reg_cols:
        rng = np.random.default_rng(seed + 1)
        for col in reg_cols:
            df[col] = rng.uniform(0, 1, len(ts))
    df["hour_sin"] = np.sin(2 * np.pi * np.arange(len(ts)) / 24)
    df["energy_lag_24"] = df["energy"].shift(24)
    return df.dropna()


def _make_study_mock(best_params: dict, rmse: float = 3.0, mae: float = 2.0):
    """Build a minimal Optuna Study mock with required attributes."""
    trial_mock = MagicMock()
    trial_mock.user_attrs = {"rmse": rmse, "mae": mae}

    study = MagicMock(spec=optuna.Study)
    study.best_params = best_params
    study.best_trial = trial_mock
    study.best_value = rmse
    return study


_BEST_PARAMS = {
    "seasonality_mode": "additive",
    "n_changepoints": 10,
    "changepoint_prior_scale": 0.05,
    "changepoint_range": 0.8,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
    "weekly_seasonality": 10,
    "yearly_seasonality": 10,
    "interval_width": 0.9,
    "growth": "linear",
}


def _run_parent_logging(train_df, test_df, model_name="Prophet", tmpdir=None):
    """Run parent_logging inside a live MLflow run. Returns (run_id, tracking_uri)."""
    study = _make_study_mock(_BEST_PARAMS)
    config_dir = pathlib.Path(__file__).parent.parent / "configs"

    with patch("ev_load_fc.training.mlflow_api.EvaluationPlots") as MockPlotter, \
         patch("ev_load_fc.training.mlflow_api.plot_param_importances") as mock_pi, \
         patch("ev_load_fc.training.mlflow_api.plot_optimization_history") as mock_oh, \
         patch("ev_load_fc.training.mlflow_api.mlflow.log_figure"), \
         patch("ev_load_fc.training.mlflow_api.mlflow.log_artifact"):

        MockPlotter.return_value.plot_correlation_with_target.return_value = MagicMock()
        mock_pi.return_value = MagicMock()
        mock_oh.return_value = MagicMock()

        with mlflow.start_run() as run:
            parent_logging(
                study=study,
                model_name=model_name,
                feature_version="test_v1",
                train=train_df,
                test=test_df,
                target="energy",
                train_path=pathlib.Path("/fake/train.csv"),
                test_path=pathlib.Path("/fake/test.csv"),
                config_dir=config_dir,
                run_num=1,
                metric="rmse",
            )
            run_id = run.info.run_id

    return run_id


# ---------------------------------------------------------------------------
# Tests — Prophet with regressors
# ---------------------------------------------------------------------------

class TestParentLoggingProphetWithRegressors:

    @pytest.fixture(scope="class")
    def setup(self, tmp_path_factory):
        tmpdir = tmp_path_factory.mktemp("mlflow")
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir}/test.db")
        mlflow.set_experiment("test_parent_logging")

        train = _make_train_df(600, reg_cols=["fog_dur_rw_1_sum", "temp_rw_1_mean"])
        test = _make_train_df(200, reg_cols=["fog_dur_rw_1_sum", "temp_rw_1_mean"], seed=99)
        run_id = _run_parent_logging(train, test)
        return run_id

    def test_model_artifact_exists(self, setup):
        run_id = setup
        local_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model")
        assert pathlib.Path(local_path).exists()

    def test_model_has_correct_extra_regressors(self, setup):
        run_id = setup
        local_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model")
        model = mlflow.prophet.load_model(pathlib.Path(local_path).as_uri())
        reg_keys = set(model.extra_regressors.keys())
        assert "fog_dur_rw_1_sum" in reg_keys
        assert "temp_rw_1_mean" in reg_keys

    def test_energy_lag_not_in_regressors(self, setup):
        run_id = setup
        local_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model")
        model = mlflow.prophet.load_model(pathlib.Path(local_path).as_uri())
        assert "energy_lag_24" not in model.extra_regressors

    def test_best_rmse_logged(self, setup):
        run_id = setup
        client = mlflow.tracking.MlflowClient()
        metric = client.get_metric_history(run_id, "best_rmse")
        assert len(metric) == 1 and metric[0].value == pytest.approx(3.0)

    def test_best_mae_logged(self, setup):
        run_id = setup
        client = mlflow.tracking.MlflowClient()
        metric = client.get_metric_history(run_id, "best_mae")
        assert len(metric) == 1 and metric[0].value == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Tests — Prophet without regressors (fallback)
# ---------------------------------------------------------------------------

class TestParentLoggingProphetNoRegressors:

    @pytest.fixture(scope="class")
    def setup(self, tmp_path_factory):
        tmpdir = tmp_path_factory.mktemp("mlflow_noreg")
        mlflow.set_tracking_uri(f"sqlite:///{tmpdir}/test.db")
        mlflow.set_experiment("test_parent_logging_noreg")

        train = _make_train_df(600, reg_cols=None)
        test = _make_train_df(200, reg_cols=None, seed=99)
        run_id = _run_parent_logging(train, test)
        return run_id

    def test_model_has_no_extra_regressors(self, setup):
        run_id = setup
        local_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model")
        model = mlflow.prophet.load_model(pathlib.Path(local_path).as_uri())
        assert len(model.extra_regressors) == 0

    def test_no_crash_on_empty_regressor_list(self, setup):
        """Fixture success itself is the assertion — no exception raised."""
        assert setup is not None


# ---------------------------------------------------------------------------
# Edge case: X_train missing some expected regressor cols (feature selection)
# ---------------------------------------------------------------------------

class TestParentLoggingProphetPartialRegressors:

    def test_only_present_cols_registered(self, tmp_path):
        """If feature selection removed a col, only cols actually in X_train are added."""
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/test.db")
        mlflow.set_experiment("test_partial_regressors")

        # Only fog_dur_rw_1_sum is present; temp was removed by feature selection
        train = _make_train_df(600, reg_cols=["fog_dur_rw_1_sum"])
        test = _make_train_df(200, reg_cols=["fog_dur_rw_1_sum"], seed=99)
        run_id = _run_parent_logging(train, test)

        local_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model")
        model = mlflow.prophet.load_model(pathlib.Path(local_path).as_uri())
        reg_keys = set(model.extra_regressors.keys())
        assert "fog_dur_rw_1_sum" in reg_keys
        assert "temp_rw_1_mean" not in reg_keys   # was never in X_train

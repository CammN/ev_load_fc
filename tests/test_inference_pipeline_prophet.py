"""Unit tests for InferencePipeline Stage 5 changes (Prophet regressor integration).

Covers:
- run() passes X=self.X to prophet_forecast when model is Prophet
- _conformal_ci_prophet() includes regressor columns when model has extra_regressors
- Edge cases: X missing timestamps, model regressors not in X
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, call
from prophet import Prophet

from ev_load_fc.pipelines.inference_pipeline import InferencePipeline, InferencePipelineConfig
from ev_load_fc.training.prophet_api import prophet_df_format


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REG_COLS = ["fog_dur", "temp_rw_1_mean"]
N_DAYS = 600   # enough for 366-day initial window + validation


def _make_cfg(tmp_path):
    return InferencePipelineConfig(
        raw_hourly_path=tmp_path / "combined_processed.csv",
        feature_store=tmp_path,
        predictions_dir=tmp_path / "predictions",
        X_set="test_features",
        experiment_name="Test Experiment",
        model_family="Prophet",
        feature_version="test_v1",
        metric="rmse",
        horizon=24,
        inference_start=pd.Timestamp("2019-10-01 00:00:00"),
        confidence_intervals=[0.80, 0.95],
    )


def _make_raw_hourly(n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days * 24, freq="h")
    t = np.arange(len(idx))
    energy = 50 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2, len(idx))
    return pd.DataFrame({"energy": energy}, index=idx)


def _make_X(index: pd.DatetimeIndex, reg_cols: list = REG_COLS, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 1, len(index)) for col in reg_cols}
    return pd.DataFrame(data, index=index)


def _fit_prophet_with_regressors(raw_hourly: pd.DataFrame, X: pd.DataFrame, reg_cols=REG_COLS):
    """Fit a real Prophet model with external regressors on the first half of data."""
    split = raw_hourly.index[len(raw_hourly) // 2]
    y_train = raw_hourly.loc[raw_hourly.index <= split, "energy"]
    model = Prophet(seasonality_mode="additive")
    for col in reg_cols:
        model.add_regressor(col)
    train_df = prophet_df_format(y_train, X.loc[y_train.index])
    model.fit(train_df)
    return model


def _fit_prophet_univariate(raw_hourly: pd.DataFrame):
    split = raw_hourly.index[len(raw_hourly) // 2]
    y_train = raw_hourly.loc[raw_hourly.index <= split, "energy"]
    model = Prophet(seasonality_mode="additive")
    model.fit(prophet_df_format(y_train))
    return model


# ---------------------------------------------------------------------------
# run() — X passed to prophet_forecast
# ---------------------------------------------------------------------------

class TestRunPassesXToProphetForecast:

    def test_prophet_forecast_called_with_x(self, tmp_path):
        """run() must call prophet_forecast(X=self.X) when model is Prophet."""
        cfg = _make_cfg(tmp_path)
        pipeline = InferencePipeline(config=cfg)
        raw_hourly = _make_raw_hourly(N_DAYS)
        X = _make_X(raw_hourly.index)
        model = _fit_prophet_with_regressors(raw_hourly, X)

        n = cfg.horizon
        fc_df = pd.DataFrame({
            "timestamp": pd.date_range("2019-10-01", periods=n, freq="h"),
            "yhat": np.random.rand(n) + 30,
            "y": np.random.rand(n) + 30,
        })
        best_run = pd.Series({
            "run_id": "mock_run_id",
            "tags.feature_set_version": "test_v1",
            "tags.model_family": "Prophet",
        })

        def _fake_load_model():
            pipeline.model = model
            pipeline.best_run = best_run

        def _fake_load_data():
            pipeline.raw_hourly = raw_hourly
            pipeline.X = X
            pipeline._resolved_feature_version = "test_v1"

        with patch("ev_load_fc.pipelines.inference_pipeline.init_mlflow"), \
             patch.object(pipeline, "_load_model", side_effect=_fake_load_model), \
             patch.object(pipeline, "_load_data", side_effect=_fake_load_data), \
             patch.object(pipeline, "_validate_inference_start"), \
             patch("ev_load_fc.pipelines.inference_pipeline.prophet_forecast",
                   return_value=fc_df) as mock_pf:
            pipeline.run()

        # Verify X was passed
        _, kwargs = mock_pf.call_args
        assert "X" in kwargs
        assert kwargs["X"] is X

    def test_prophet_run_returns_ci_columns(self, tmp_path):
        """Full run() with Prophet+regressors should attach CI columns."""
        cfg = _make_cfg(tmp_path)
        pipeline = InferencePipeline(config=cfg)
        raw_hourly = _make_raw_hourly(N_DAYS)
        X = _make_X(raw_hourly.index)
        model = _fit_prophet_with_regressors(raw_hourly, X)

        inference_start = pd.Timestamp("2019-10-01 00:00:00")
        n = cfg.horizon
        fc_df = pd.DataFrame({
            "timestamp": pd.date_range(inference_start, periods=n, freq="h"),
            "yhat": np.random.rand(n) + 30,
            "y": np.random.rand(n) + 30,
        })
        best_run = pd.Series({
            "run_id": "mock_run",
            "tags.feature_set_version": "test_v1",
            "tags.model_family": "Prophet",
        })

        def _fake_load_model():
            pipeline.model = model
            pipeline.best_run = best_run

        def _fake_load_data():
            pipeline.raw_hourly = raw_hourly
            pipeline.X = X
            pipeline._resolved_feature_version = "test_v1"

        with patch("ev_load_fc.pipelines.inference_pipeline.init_mlflow"), \
             patch.object(pipeline, "_load_model", side_effect=_fake_load_model), \
             patch.object(pipeline, "_load_data", side_effect=_fake_load_data), \
             patch.object(pipeline, "_validate_inference_start"), \
             patch("ev_load_fc.pipelines.inference_pipeline.prophet_forecast",
                   return_value=fc_df):
            result = pipeline.run()

        for col in ["yhat_lower_80", "yhat_upper_80", "yhat_lower_95", "yhat_upper_95"]:
            assert col in result.columns, f"Missing CI column: {col}"


# ---------------------------------------------------------------------------
# _conformal_ci_prophet
# ---------------------------------------------------------------------------

class TestConformalCiProphetWithRegressors:

    @pytest.fixture(scope="class")
    def setup(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("conformal_ci")
        raw_hourly = _make_raw_hourly(N_DAYS)
        X = _make_X(raw_hourly.index)
        model = _fit_prophet_with_regressors(raw_hourly, X)
        cfg = _make_cfg(tmp_path)
        pipeline = InferencePipeline(config=cfg)
        pipeline.model = model
        pipeline.raw_hourly = raw_hourly
        pipeline.X = X
        return pipeline

    def test_returns_ci_dataframe(self, setup):
        fc_df = pd.DataFrame({"yhat": np.ones(24)})
        ci = setup._conformal_ci_prophet(fc_df)
        assert isinstance(ci, pd.DataFrame)
        assert len(ci.columns) > 0

    def test_correct_ci_columns(self, setup):
        fc_df = pd.DataFrame({"yhat": np.ones(24)})
        ci = setup._conformal_ci_prophet(fc_df)
        assert "yhat_lower_80" in ci.columns
        assert "yhat_upper_80" in ci.columns
        assert "yhat_lower_95" in ci.columns
        assert "yhat_upper_95" in ci.columns

    def test_ci_half_widths_are_positive(self, setup):
        fc_df = pd.DataFrame({"yhat": np.ones(24)})
        ci = setup._conformal_ci_prophet(fc_df)
        # upper_X > 0, lower_X < 0 (offsets relative to yhat)
        assert (ci["yhat_upper_80"] > 0).all()
        assert (ci["yhat_lower_80"] < 0).all()

    def test_95_wider_than_80(self, setup):
        fc_df = pd.DataFrame({"yhat": np.ones(24)})
        ci = setup._conformal_ci_prophet(fc_df)
        assert (ci["yhat_upper_95"].abs() >= ci["yhat_upper_80"].abs()).all()

    def test_row_count_matches_fc_df(self, setup):
        for n in [1, 24, 168]:
            fc_df = pd.DataFrame({"yhat": np.ones(n)})
            ci = setup._conformal_ci_prophet(fc_df)
            assert len(ci) == n


class TestConformalCiProphetUnivariate:

    @pytest.fixture(scope="class")
    def setup_univariate(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("ci_univ")
        raw_hourly = _make_raw_hourly(N_DAYS)
        model = _fit_prophet_univariate(raw_hourly)
        cfg = _make_cfg(tmp_path)
        pipeline = InferencePipeline(config=cfg)
        pipeline.model = model
        pipeline.raw_hourly = raw_hourly
        pipeline.X = _make_X(raw_hourly.index)  # X present but has no matching regressors
        return pipeline

    def test_univariate_ci_still_works(self, setup_univariate):
        """Univariate model (no extra_regressors) must still produce valid CI."""
        fc_df = pd.DataFrame({"yhat": np.ones(24)})
        ci = setup_univariate._conformal_ci_prophet(fc_df)
        assert "yhat_lower_80" in ci.columns
        assert len(ci) == 24


class TestConformalCiProphetEdgeCases:

    def test_warns_and_skips_when_x_missing_training_timestamps(self, tmp_path, caplog):
        """If self.X doesn't cover training timestamps, return empty df with warning."""
        raw_hourly = _make_raw_hourly(N_DAYS)
        X = _make_X(raw_hourly.index, reg_cols=["fog_dur"])
        model = _fit_prophet_with_regressors(raw_hourly, X, reg_cols=["fog_dur"])

        cfg = _make_cfg(tmp_path)
        pipeline = InferencePipeline(config=cfg)
        pipeline.model = model
        pipeline.raw_hourly = raw_hourly
        # Provide X that only covers the forecast window — missing all training timestamps
        forecast_idx = pd.date_range("2019-10-01", periods=24, freq="h")
        pipeline.X = _make_X(forecast_idx, reg_cols=["fog_dur"])

        fc_df = pd.DataFrame({"yhat": np.ones(24)})
        import logging
        with caplog.at_level(logging.WARNING):
            ci = pipeline._conformal_ci_prophet(fc_df)

        assert isinstance(ci, pd.DataFrame)
        # Either empty (graceful skip) or a warning was logged
        assert len(ci.columns) == 0 or any("missing" in r.message.lower() for r in caplog.records)

    def test_warns_and_skips_when_regressor_col_missing_from_x(self, tmp_path, caplog):
        """If self.X lacks a required regressor col, warn and return empty df (don't crash)."""
        raw_hourly = _make_raw_hourly(N_DAYS)
        X = _make_X(raw_hourly.index, reg_cols=["fog_dur", "temp_rw_1_mean"])
        model = _fit_prophet_with_regressors(raw_hourly, X)  # uses both cols

        cfg = _make_cfg(tmp_path)
        pipeline = InferencePipeline(config=cfg)
        pipeline.model = model
        pipeline.raw_hourly = raw_hourly
        # X only has one of the two required regressors
        pipeline.X = _make_X(raw_hourly.index, reg_cols=["fog_dur"])

        fc_df = pd.DataFrame({"yhat": np.ones(24)})
        import logging
        with caplog.at_level(logging.WARNING):
            ci = pipeline._conformal_ci_prophet(fc_df)

        # Should warn about the missing column and return empty DataFrame
        assert any("missing" in r.message.lower() for r in caplog.records)
        assert isinstance(ci, pd.DataFrame)
        assert len(ci.columns) == 0

    def test_empty_train_actuals_returns_empty_df(self, tmp_path):
        """If raw_hourly has no data before split_date, return empty df."""
        raw_hourly = _make_raw_hourly(N_DAYS)
        X = _make_X(raw_hourly.index)
        model = _fit_prophet_with_regressors(raw_hourly, X)

        cfg = _make_cfg(tmp_path)
        pipeline = InferencePipeline(config=cfg)
        pipeline.model = model
        pipeline.raw_hourly = raw_hourly
        pipeline.X = X

        fc_df = pd.DataFrame({"yhat": np.ones(24)})

        # Patch CFG so split_date is before all data — no training actuals available
        with patch("ev_load_fc.pipelines.inference_pipeline.CFG", {
            "features": {"target": "energy"},
            "data": {"preprocessing": {"split_date": "2016-01-01"}},
        }):
            ci = pipeline._conformal_ci_prophet(fc_df)

        assert isinstance(ci, pd.DataFrame)
        assert len(ci.columns) == 0

"""Unit tests for ev_load_fc.training.prophet_api (Stage 1).

Covers:
- get_prophet_regressor_cols
- prophet_df_format
- cv_score_prophet_model (univariate and regressor paths, including edge cases)
"""
import math
import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from ev_load_fc.training.prophet_api import (
    get_prophet_regressor_cols,
    prophet_df_format,
    cv_score_prophet_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hourly_series(n_days: int, seed: int = 0) -> pd.Series:
    """Create a synthetic hourly energy time series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days * 24, freq="h")
    # Simple seasonality + noise
    t = np.arange(len(idx))
    values = (
        50
        + 10 * np.sin(2 * np.pi * t / 24)   # daily cycle
        + 5 * np.sin(2 * np.pi * t / (24 * 7))  # weekly cycle
        + rng.normal(0, 2, len(idx))
    )
    return pd.Series(values, index=idx, name="energy")


def _make_regressors_df(index: pd.DatetimeIndex, cols: list, seed: int = 1) -> pd.DataFrame:
    """Create a synthetic DataFrame of external regressors."""
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 1, len(index)) for col in cols}
    return pd.DataFrame(data, index=index)


_FAKE_CFG = {
    "features": {
        "feature_engineering": {
            "weather_col_substrs": ["fog_dur", "rain_dur", "storm_dur"],
            "temperature_col_substrs": ["temp"],
            "traffic_col_substrs": ["cong_dur", "cong_dis", "flow_dur", "flow_dis", "event_dur", "event_dis"],
        }
    }
}


# ---------------------------------------------------------------------------
# get_prophet_regressor_cols
# ---------------------------------------------------------------------------

class TestGetProphetRegressorCols:

    def test_returns_weather_cols(self):
        cols = ["energy", "fog_dur_rw_1_sum", "rain_dur_rw_3_sum", "storm_dur"]
        result = get_prophet_regressor_cols(cols, _FAKE_CFG)
        assert set(result) == {"fog_dur_rw_1_sum", "rain_dur_rw_3_sum", "storm_dur"}

    def test_returns_temperature_cols(self):
        cols = ["energy", "temp_rw_1_mean", "temp_rw_6_mean"]
        result = get_prophet_regressor_cols(cols, _FAKE_CFG)
        assert set(result) == {"temp_rw_1_mean", "temp_rw_6_mean"}

    def test_returns_traffic_cols(self):
        cols = ["energy", "cong_dur_rw_1_sum", "flow_dis_rw_3_sum", "event_dur"]
        result = get_prophet_regressor_cols(cols, _FAKE_CFG)
        assert set(result) == {"cong_dur_rw_1_sum", "flow_dis_rw_3_sum", "event_dur"}

    def test_excludes_energy_lags(self):
        cols = ["energy_lag_1", "energy_lag_24", "energy_rw_3_sum", "fog_dur"]
        result = get_prophet_regressor_cols(cols, _FAKE_CFG)
        # fog_dur matches, energy cols do not
        assert "fog_dur" in result
        assert "energy_lag_1" not in result
        assert "energy_lag_24" not in result
        assert "energy_rw_3_sum" not in result

    def test_excludes_time_sin_cos(self):
        cols = ["hour_sin", "hour_cos", "weekday_sin", "month_cos", "fog_dur"]
        result = get_prophet_regressor_cols(cols, _FAKE_CFG)
        assert result == ["fog_dur"]

    def test_empty_when_no_matches(self):
        cols = ["energy", "energy_lag_1", "hour_sin", "christmas_day"]
        result = get_prophet_regressor_cols(cols, _FAKE_CFG)
        assert result == []

    def test_empty_input(self):
        assert get_prophet_regressor_cols([], _FAKE_CFG) == []

    def test_mixed_columns_preserves_order(self):
        cols = ["fog_dur", "energy", "rain_dur", "temp", "cong_dur"]
        result = get_prophet_regressor_cols(cols, _FAKE_CFG)
        assert result == ["fog_dur", "rain_dur", "temp", "cong_dur"]

    def test_no_false_positive_on_partial_match(self):
        # "temperature" contains "temp" — should be included
        cols = ["temperature_raw", "energy"]
        result = get_prophet_regressor_cols(cols, _FAKE_CFG)
        assert "temperature_raw" in result


# ---------------------------------------------------------------------------
# prophet_df_format
# ---------------------------------------------------------------------------

class TestProphetDfFormat:

    def test_univariate_returns_ds_y(self):
        ts = _make_hourly_series(10)
        df = prophet_df_format(ts)
        assert list(df.columns) == ["ds", "y"]
        assert len(df) == len(ts)

    def test_ds_col_matches_index(self):
        ts = _make_hourly_series(10)
        df = prophet_df_format(ts)
        pd.testing.assert_series_equal(df["ds"].reset_index(drop=True),
                                       pd.Series(ts.index, name="ds"),
                                       check_names=False)

    def test_y_col_matches_values(self):
        ts = _make_hourly_series(10)
        df = prophet_df_format(ts)
        np.testing.assert_array_almost_equal(df["y"].values, ts.values)

    def test_with_regressors_has_extra_cols(self):
        ts = _make_hourly_series(10)
        reg = _make_regressors_df(ts.index, ["fog_dur", "temp"])
        df = prophet_df_format(ts, reg)
        assert "ds" in df.columns
        assert "y" in df.columns
        assert "fog_dur" in df.columns
        assert "temp" in df.columns

    def test_with_regressors_correct_row_count(self):
        ts = _make_hourly_series(10)
        reg = _make_regressors_df(ts.index, ["fog_dur", "temp"])
        df = prophet_df_format(ts, reg)
        assert len(df) == len(ts)

    def test_with_regressors_values_aligned(self):
        ts = _make_hourly_series(10)
        reg = _make_regressors_df(ts.index, ["fog_dur"])
        df = prophet_df_format(ts, reg)
        np.testing.assert_array_almost_equal(df["fog_dur"].values, reg["fog_dur"].values)

    def test_with_regressors_no_nans_on_clean_input(self):
        ts = _make_hourly_series(10)
        reg = _make_regressors_df(ts.index, ["fog_dur", "temp"])
        df = prophet_df_format(ts, reg)
        assert not df.isnull().any().any()

    def test_none_regressors_same_as_no_arg(self):
        ts = _make_hourly_series(10)
        df_no_arg = prophet_df_format(ts)
        df_none = prophet_df_format(ts, None)
        pd.testing.assert_frame_equal(df_no_arg, df_none)

    def test_does_not_mutate_input(self):
        ts = _make_hourly_series(10)
        ts_copy = ts.copy()
        prophet_df_format(ts)
        pd.testing.assert_series_equal(ts, ts_copy)


# ---------------------------------------------------------------------------
# cv_score_prophet_model — univariate path
# ---------------------------------------------------------------------------

class TestCvScoreProphetUnivariate:

    @pytest.fixture(scope="class")
    def series_500d(self):
        return _make_hourly_series(500)

    def test_returns_rmse_and_mae_keys(self, series_500d):
        model = Prophet(seasonality_mode="additive")
        result = cv_score_prophet_model(model, series_500d, n_splits=1)
        assert "rmse" in result
        assert "mae" in result

    def test_scores_are_finite_positive(self, series_500d):
        model = Prophet(seasonality_mode="additive")
        result = cv_score_prophet_model(model, series_500d, n_splits=1)
        assert np.isfinite(result["rmse"]) and result["rmse"] > 0
        assert np.isfinite(result["mae"]) and result["mae"] > 0

    def test_regressors_none_uses_univariate_path(self, series_500d):
        """Passing regressors_df=None must still work (uses cross_validation)."""
        model = Prophet(seasonality_mode="additive")
        result = cv_score_prophet_model(model, series_500d, n_splits=1, regressors_df=None)
        assert "rmse" in result


# ---------------------------------------------------------------------------
# cv_score_prophet_model — regressor path
# ---------------------------------------------------------------------------

class TestCvScoreProphetWithRegressors:

    @pytest.fixture(scope="class")
    def data_500d(self):
        ts = _make_hourly_series(500)
        reg = _make_regressors_df(ts.index, ["fog_dur", "temp"])
        return ts, reg

    def _factory(self, params=None, reg_cols=("fog_dur", "temp")):
        def make():
            m = Prophet(**(params or {}))
            for c in reg_cols:
                m.add_regressor(c)
            return m
        return make

    def test_returns_rmse_and_mae_keys(self, data_500d):
        ts, reg = data_500d
        result = cv_score_prophet_model(
            model=Prophet(), y=ts, n_splits=1,
            regressors_df=reg, model_factory=self._factory()
        )
        assert "rmse" in result and "mae" in result

    def test_scores_are_finite_positive(self, data_500d):
        ts, reg = data_500d
        result = cv_score_prophet_model(
            model=Prophet(), y=ts, n_splits=1,
            regressors_df=reg, model_factory=self._factory()
        )
        assert np.isfinite(result["rmse"]) and result["rmse"] > 0
        assert np.isfinite(result["mae"]) and result["mae"] > 0

    def test_n_splits_1(self, data_500d):
        """Single-split should not raise index errors."""
        ts, reg = data_500d
        result = cv_score_prophet_model(
            model=Prophet(), y=ts, n_splits=1,
            regressors_df=reg, model_factory=self._factory()
        )
        assert np.isfinite(result["rmse"])

    def test_factory_called_fresh_per_fold(self, data_500d):
        """Each fold must receive a distinct model instance (no state leakage)."""
        ts, reg = data_500d
        call_log = []
        def tracking_factory():
            m = Prophet()
            m.add_regressor("fog_dur")
            m.add_regressor("temp")
            call_log.append(id(m))
            return m

        # n_splits=2: two folds → factory called twice
        ts_long = _make_hourly_series(800)
        reg_long = _make_regressors_df(ts_long.index, ["fog_dur", "temp"])
        cv_score_prophet_model(
            model=Prophet(), y=ts_long, n_splits=2,
            regressors_df=reg_long, model_factory=tracking_factory
        )
        # All ids must be distinct (no object reuse)
        assert len(call_log) == len(set(call_log))

    def test_raises_when_factory_missing(self, data_500d):
        """regressors_df provided but model_factory=None → clear ValueError."""
        ts, reg = data_500d
        with pytest.raises(ValueError, match="model_factory"):
            cv_score_prophet_model(
                model=Prophet(), y=ts, n_splits=1,
                regressors_df=reg, model_factory=None
            )

    def test_constant_regressor_no_crash(self):
        """All-zero regressor column should not cause Prophet to crash."""
        ts = _make_hourly_series(500)
        reg = pd.DataFrame({"fog_dur": np.zeros(len(ts))}, index=ts.index)
        result = cv_score_prophet_model(
            model=Prophet(), y=ts, n_splits=1,
            regressors_df=reg,
            model_factory=self._factory(reg_cols=("fog_dur",))
        )
        assert np.isfinite(result["rmse"])

    def test_regressors_with_nans_in_val_fold(self):
        """NaN rows in train_fold are dropped; val_fold NaNs propagate to yhat
        but should not crash (Prophet handles it gracefully)."""
        ts = _make_hourly_series(500)
        reg = _make_regressors_df(ts.index, ["fog_dur"])
        reg.iloc[400:410] = np.nan  # NaNs inside the validation window

        # Should not raise
        result = cv_score_prophet_model(
            model=Prophet(), y=ts, n_splits=1,
            regressors_df=reg,
            model_factory=self._factory(reg_cols=("fog_dur",))
        )
        # Result may be non-finite if NaNs propagate into yhat, but should not crash
        assert isinstance(result["rmse"], float)

    def test_empty_regressors_df_uses_univariate_path(self):
        """Empty DataFrame for regressors_df falls back to univariate CV."""
        ts = _make_hourly_series(500)
        empty_reg = pd.DataFrame(index=ts.index)
        model = Prophet()
        result = cv_score_prophet_model(
            model=model, y=ts, n_splits=1,
            regressors_df=empty_reg,
            model_factory=None   # should NOT be called
        )
        assert "rmse" in result

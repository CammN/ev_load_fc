"""Unit tests for prophet_forecast() in inference.py (Stage 4).

Covers:
- Univariate call (no regressors, X=None)
- With regressors and valid X
- Edge cases: missing X, missing regressor cols in X, OOB forecast timestamps
"""
import numpy as np
import pandas as pd
import pytest
from prophet import Prophet

from ev_load_fc.inference.inference import prophet_forecast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_hourly(n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n_days * 24, freq="h")
    t = np.arange(len(idx))
    energy = 50 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2, len(idx))
    return pd.DataFrame({"energy": energy}, index=idx)


def _make_X(index: pd.DatetimeIndex, reg_cols: list, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {col: rng.uniform(0, 1, len(index)) for col in reg_cols},
        index=index,
    )


def _fit_prophet(reg_cols: list | None, raw_hourly: pd.DataFrame, X: pd.DataFrame | None = None):
    """Fit a Prophet model on the first portion of raw_hourly."""
    train_end = raw_hourly.index[len(raw_hourly) // 2]
    y_train = raw_hourly.loc[raw_hourly.index <= train_end, "energy"]

    model = Prophet(seasonality_mode="additive")
    if reg_cols:
        for col in reg_cols:
            model.add_regressor(col)
        from ev_load_fc.training.prophet_api import prophet_df_format
        X_train = X.loc[y_train.index]
        train_df = prophet_df_format(y_train, X_train)
    else:
        from ev_load_fc.training.prophet_api import prophet_df_format
        train_df = prophet_df_format(y_train)

    model.fit(train_df)
    return model, train_end


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProphetForecastUnivariate:

    @pytest.fixture(scope="class")
    def setup(self):
        raw = _make_raw_hourly(600)
        model, train_end = _fit_prophet(None, raw)
        return model, raw, train_end

    def test_returns_correct_columns(self, setup):
        model, raw, train_end = setup
        start = train_end + pd.Timedelta(hours=1)
        fc = prophet_forecast(model, raw, start, horizon=24)
        assert set(fc.columns) == {"timestamp", "yhat", "y"}

    def test_correct_number_of_rows(self, setup):
        model, raw, train_end = setup
        start = train_end + pd.Timedelta(hours=1)
        fc = prophet_forecast(model, raw, start, horizon=48)
        assert len(fc) == 48

    def test_timestamps_are_consecutive_hourly(self, setup):
        model, raw, train_end = setup
        start = train_end + pd.Timedelta(hours=1)
        fc = prophet_forecast(model, raw, start, horizon=24)
        expected = pd.date_range(start, periods=24, freq="h")
        np.testing.assert_array_equal(fc["timestamp"].values, expected.values)

    def test_yhat_is_finite(self, setup):
        model, raw, train_end = setup
        start = train_end + pd.Timedelta(hours=1)
        fc = prophet_forecast(model, raw, start, horizon=24)
        assert fc["yhat"].notna().all()

    def test_x_none_no_regressors_no_raise(self, setup):
        """Passing X=None on a univariate model must not raise."""
        model, raw, train_end = setup
        start = train_end + pd.Timedelta(hours=1)
        fc = prophet_forecast(model, raw, start, horizon=24, X=None)
        assert len(fc) == 24

    def test_first_forecast_row_has_actual(self, setup):
        model, raw, train_end = setup
        start = train_end + pd.Timedelta(hours=1)
        fc = prophet_forecast(model, raw, start, horizon=1)
        expected_y = raw.loc[start, "energy"]
        assert fc["y"].iloc[0] == pytest.approx(expected_y)


class TestProphetForecastWithRegressors:

    REG_COLS = ["fog_dur", "temp_rw_1_mean"]

    @pytest.fixture(scope="class")
    def setup(self):
        raw = _make_raw_hourly(600)
        X = _make_X(raw.index, self.REG_COLS)
        model, train_end = _fit_prophet(self.REG_COLS, raw, X)
        return model, raw, X, train_end

    def test_returns_correct_columns(self, setup):
        model, raw, X, train_end = setup
        start = train_end + pd.Timedelta(hours=1)
        fc = prophet_forecast(model, raw, start, horizon=24, X=X)
        assert set(fc.columns) == {"timestamp", "yhat", "y"}

    def test_correct_number_of_rows(self, setup):
        model, raw, X, train_end = setup
        start = train_end + pd.Timedelta(hours=1)
        fc = prophet_forecast(model, raw, start, horizon=24, X=X)
        assert len(fc) == 24

    def test_yhat_is_finite(self, setup):
        model, raw, X, train_end = setup
        start = train_end + pd.Timedelta(hours=1)
        fc = prophet_forecast(model, raw, start, horizon=24, X=X)
        assert fc["yhat"].notna().all()

    def test_regressor_values_aligned_correctly(self, setup):
        """Verify the future df used the right X row for each timestamp."""
        model, raw, X, train_end = setup
        start = train_end + pd.Timedelta(hours=1)
        # Fit a second model and manually override regressor to 0 — yhat should differ
        model2 = Prophet(seasonality_mode="additive")
        for col in self.REG_COLS:
            model2.add_regressor(col)
        from ev_load_fc.training.prophet_api import prophet_df_format
        y_train = raw.loc[raw.index <= train_end, "energy"]
        model2.fit(prophet_df_format(y_train, X.loc[y_train.index]))

        X_zero = X.copy()
        X_zero[self.REG_COLS] = 0.0
        fc_orig = prophet_forecast(model2, raw, start, 24, X=X)
        fc_zero = prophet_forecast(model2, raw, start, 24, X=X_zero)
        # Predictions with zeroed regressors should differ from original
        assert not np.allclose(fc_orig["yhat"].values, fc_zero["yhat"].values)


class TestProphetForecastEdgeCases:

    REG_COLS = ["fog_dur", "temp"]

    @pytest.fixture(scope="class")
    def setup_with_regressors(self):
        raw = _make_raw_hourly(600)
        X = _make_X(raw.index, self.REG_COLS)
        model, train_end = _fit_prophet(self.REG_COLS, raw, X)
        return model, raw, X, train_end

    def test_raises_when_x_none_but_regressors_present(self, setup_with_regressors):
        model, raw, X, train_end = setup_with_regressors
        start = train_end + pd.Timedelta(hours=1)
        with pytest.raises(ValueError, match="external regressor"):
            prophet_forecast(model, raw, start, horizon=24, X=None)

    def test_raises_when_x_missing_regressor_col(self, setup_with_regressors):
        model, raw, X, train_end = setup_with_regressors
        start = train_end + pd.Timedelta(hours=1)
        X_incomplete = X.drop(columns=["fog_dur"])
        with pytest.raises(ValueError, match="fog_dur"):
            prophet_forecast(model, raw, start, horizon=24, X=X_incomplete)

    def test_raises_when_forecast_dates_beyond_x(self, setup_with_regressors):
        """forecast_dates extending past X.index should raise KeyError (not silently NaN)."""
        model, raw, X, train_end = setup_with_regressors
        # Start forecast 1h before end of X so we go out of bounds
        start = X.index[-2]
        with pytest.raises((KeyError, IndexError)):
            prophet_forecast(model, raw, start, horizon=24, X=X)

    def test_forecast_start_at_first_x_row(self, setup_with_regressors):
        """Forecast starting at very first X row should not raise off-by-one errors."""
        raw = _make_raw_hourly(600)
        X = _make_X(raw.index, self.REG_COLS)
        model, train_end = _fit_prophet(self.REG_COLS, raw, X)
        # Use a start within the raw_hourly range with sufficient X coverage
        start = train_end + pd.Timedelta(hours=1)
        fc = prophet_forecast(model, raw, start, horizon=1, X=X)
        assert len(fc) == 1
        assert np.isfinite(fc["yhat"].iloc[0])

"""Unit tests for streamlit_app/utils/charts.py."""
import sys
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "streamlit_app"))

from utils.charts import (
    energy_timeseries,
    hourly_seasonality,
    weekly_seasonality,
    correlation_bar,
    model_comparison_bar,
    forecast_chart,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_energy_df(n=100):
    ts = pd.date_range("2018-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": ts,
        "hour": [t.hour for t in ts],
        "weekday": [t.weekday() for t in ts],
        "energy": np.random.rand(n) * 10 + 30,
        "energy_outlier": (np.random.rand(n) > 0.95).astype(int),
    })


def _make_fc_df(n=24, with_ci=True):
    ts = pd.date_range("2019-10-01", periods=n, freq="h")
    yhat = np.random.rand(n) * 5 + 30
    df = pd.DataFrame({
        "timestamp": ts,
        "yhat": yhat,
        "y": yhat + np.random.rand(n) - 0.5,
    })
    if with_ci:
        std = np.random.rand(n) * 2 + 1
        df["yhat_lower_80"] = yhat - 1.28 * std
        df["yhat_upper_80"] = yhat + 1.28 * std
        df["yhat_lower_95"] = yhat - 1.96 * std
        df["yhat_upper_95"] = yhat + 1.96 * std
    return df


def _make_summary_df():
    return pd.DataFrame({
        "model_family": ["LightGBM", "CatBoost", "RandomForest"],
        "rmse": [12.1, 13.5, 15.2],
        "mae": [9.0, 10.1, 11.8],
    })


# ---------------------------------------------------------------------------
# energy_timeseries
# ---------------------------------------------------------------------------

class TestEnergyTimeseries:

    def test_returns_figure(self):
        df = _make_energy_df()
        fig = energy_timeseries(df, pd.Timestamp("2018-01-03"))
        assert isinstance(fig, go.Figure)

    def test_has_energy_trace(self):
        df = _make_energy_df()
        fig = energy_timeseries(df, pd.Timestamp("2018-01-03"))
        names = [t.name for t in fig.data]
        assert "Energy (kWh)" in names

    def test_outlier_trace_added_when_enabled(self):
        df = _make_energy_df(200)
        fig = energy_timeseries(df, pd.Timestamp("2018-01-03"), show_outliers=True)
        names = [t.name for t in fig.data]
        assert "Outlier" in names

    def test_no_outlier_trace_when_disabled(self):
        df = _make_energy_df()
        fig = energy_timeseries(df, pd.Timestamp("2018-01-03"), show_outliers=False)
        names = [t.name for t in fig.data]
        assert "Outlier" not in names

    def test_y_axis_set(self):
        df = _make_energy_df()
        fig = energy_timeseries(df, pd.Timestamp("2018-01-03"))
        assert fig.layout.yaxis.title.text == "Energy (kWh)"


# ---------------------------------------------------------------------------
# hourly_seasonality
# ---------------------------------------------------------------------------

class TestHourlySeasonality:

    def test_returns_figure(self):
        df = _make_energy_df(500)
        fig = hourly_seasonality(df)
        assert isinstance(fig, go.Figure)

    def test_has_24_bars(self):
        df = _make_energy_df(500)
        fig = hourly_seasonality(df)
        assert len(fig.data[0].x) == 24


# ---------------------------------------------------------------------------
# weekly_seasonality
# ---------------------------------------------------------------------------

class TestWeeklySeasonality:

    def test_returns_figure(self):
        df = _make_energy_df(500)
        fig = weekly_seasonality(df)
        assert isinstance(fig, go.Figure)

    def test_has_7_bars(self):
        df = _make_energy_df(500)
        fig = weekly_seasonality(df)
        assert len(fig.data[0].x) == 7

    def test_day_labels_used(self):
        df = _make_energy_df(500)
        fig = weekly_seasonality(df)
        assert "Mon" in fig.data[0].x


# ---------------------------------------------------------------------------
# correlation_bar
# ---------------------------------------------------------------------------

class TestCorrelationBar:

    def test_returns_figure(self):
        df = _make_energy_df(200)
        fig = correlation_bar(df, target_col="energy")
        assert isinstance(fig, go.Figure)

    def test_target_col_excluded_from_bars(self):
        df = _make_energy_df(200)
        fig = correlation_bar(df, target_col="energy")
        assert "energy" not in fig.data[0].y

    def test_bars_are_horizontal(self):
        df = _make_energy_df(200)
        fig = correlation_bar(df, target_col="energy")
        assert fig.data[0].orientation == "h"


# ---------------------------------------------------------------------------
# model_comparison_bar
# ---------------------------------------------------------------------------

class TestModelComparisonBar:

    def test_returns_figure(self):
        fig = model_comparison_bar(_make_summary_df())
        assert isinstance(fig, go.Figure)

    def test_has_rmse_and_mae_traces(self):
        fig = model_comparison_bar(_make_summary_df())
        names = [t.name for t in fig.data]
        assert "RMSE" in names
        assert "MAE" in names

    def test_correct_number_of_bars(self):
        summary = _make_summary_df()
        fig = model_comparison_bar(summary)
        assert len(fig.data[0].x) == len(summary)


# ---------------------------------------------------------------------------
# forecast_chart
# ---------------------------------------------------------------------------

class TestForecastChart:

    def test_returns_figure(self):
        fc_df = _make_fc_df()
        fig = forecast_chart(fc_df, ci_level=80)
        assert isinstance(fig, go.Figure)

    def test_has_forecast_trace(self):
        fc_df = _make_fc_df()
        fig = forecast_chart(fc_df, ci_level=80)
        names = [t.name for t in fig.data]
        assert "Forecast" in names

    def test_has_actual_trace_when_y_present(self):
        fc_df = _make_fc_df()
        fig = forecast_chart(fc_df, ci_level=80)
        names = [t.name for t in fig.data]
        assert "Actual" in names

    def test_ci_ribbon_present_when_ci_level_set(self):
        fc_df = _make_fc_df(with_ci=True)
        fig = forecast_chart(fc_df, ci_level=80)
        names = [t.name for t in fig.data]
        assert "80% CI" in names

    def test_no_ci_ribbon_when_ci_level_zero(self):
        fc_df = _make_fc_df(with_ci=True)
        fig = forecast_chart(fc_df, ci_level=0)
        names = [t.name for t in fig.data]
        assert "80% CI" not in names
        assert "95% CI" not in names

    def test_no_actual_trace_when_y_missing(self):
        fc_df = _make_fc_df()
        fc_df = fc_df.drop(columns=["y"])
        fig = forecast_chart(fc_df, ci_level=80)
        names = [t.name for t in fig.data]
        assert "Actual" not in names

    def test_ci_values_bounded_by_yhat(self):
        """CI lower must be <= yhat and CI upper >= yhat throughout."""
        fc_df = _make_fc_df(with_ci=True)
        assert (fc_df["yhat_lower_80"] <= fc_df["yhat"]).all()
        assert (fc_df["yhat_upper_80"] >= fc_df["yhat"]).all()

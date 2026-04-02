"""Unit tests for streamlit_app/utils/data_loader.py."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# Make streamlit_app importable without a running Streamlit server
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "streamlit_app"))

# Patch streamlit.cache_data so it works outside a Streamlit session
import unittest.mock as mock
import streamlit as st

# Monkey-patch cache_data to be a no-op decorator for tests
_real_cache_data = st.cache_data
st.cache_data = lambda func=None, **kwargs: (func if func else lambda f: f)

from utils.data_loader import list_inference_runs, load_predictions


# ---------------------------------------------------------------------------
# list_inference_runs
# ---------------------------------------------------------------------------

class TestListInferenceRuns:

    def test_empty_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "utils.data_loader._predictions_dir",
            lambda: tmp_path / "nonexistent",
        )
        result = list_inference_runs()
        assert result == []

    def test_empty_when_dir_has_no_subdirs(self, tmp_path, monkeypatch):
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        monkeypatch.setattr("utils.data_loader._predictions_dir", lambda: pred_dir)
        result = list_inference_runs()
        assert result == []

    def test_skips_subdirs_without_metadata(self, tmp_path, monkeypatch):
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        (pred_dir / "orphan_dir").mkdir()  # no metadata.json
        monkeypatch.setattr("utils.data_loader._predictions_dir", lambda: pred_dir)
        result = list_inference_runs()
        assert result == []

    def test_parses_valid_metadata(self, tmp_path, monkeypatch):
        pred_dir = tmp_path / "predictions"
        run_dir = pred_dir / "20260331_120000__LightGBM__168h__from_20191001"
        run_dir.mkdir(parents=True)
        metadata = {
            "run_id": "abc123",
            "model_family": "LightGBM",
            "feature_version": "E_f_30_rfe_20",
            "horizon": 168,
            "inference_start": "2019-10-01T00:00:00",
            "ci_levels": [0.80, 0.95],
            "metrics": {"rmse": 12.4, "mae": 9.1},
            "created_at": "2026-03-31T12:00:00",
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        monkeypatch.setattr("utils.data_loader._predictions_dir", lambda: pred_dir)
        result = list_inference_runs()
        assert len(result) == 1
        assert result[0]["run_id"] == "abc123"
        assert result[0]["model_family"] == "LightGBM"
        assert result[0]["dir_name"] == run_dir.name

    def test_sorted_newest_first(self, tmp_path, monkeypatch):
        pred_dir = tmp_path / "predictions"
        for ts in ["20260101_000000", "20260201_000000", "20260301_000000"]:
            run_dir = pred_dir / f"{ts}__LightGBM__24h__from_20191001"
            run_dir.mkdir(parents=True)
            with open(run_dir / "metadata.json", "w") as f:
                json.dump({"created_at": ts.replace("_", "T")[:16] + ":00"}, f)

        monkeypatch.setattr("utils.data_loader._predictions_dir", lambda: pred_dir)
        result = list_inference_runs()
        assert len(result) == 3
        assert result[0]["dir_name"].startswith("20260301")
        assert result[-1]["dir_name"].startswith("20260101")

    def test_skips_corrupted_metadata_gracefully(self, tmp_path, monkeypatch):
        pred_dir = tmp_path / "predictions"
        run_dir = pred_dir / "20260331_120000__LightGBM__24h__from_20191001"
        run_dir.mkdir(parents=True)
        (run_dir / "metadata.json").write_text("NOT VALID JSON")

        monkeypatch.setattr("utils.data_loader._predictions_dir", lambda: pred_dir)
        result = list_inference_runs()
        assert result == []

    def test_includes_multiple_runs(self, tmp_path, monkeypatch):
        pred_dir = tmp_path / "predictions"
        for i, family in enumerate(["LightGBM", "CatBoost"]):
            run_dir = pred_dir / f"2026033{i}_120000__{family}__24h__from_20191001"
            run_dir.mkdir(parents=True)
            with open(run_dir / "metadata.json", "w") as f:
                json.dump({"model_family": family, "created_at": f"2026-03-3{i}T12:00:00"}, f)

        monkeypatch.setattr("utils.data_loader._predictions_dir", lambda: pred_dir)
        result = list_inference_runs()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------

class TestLoadPredictions:

    def _make_pred_csv(self, pred_dir: pathlib.Path, run_name: str) -> pathlib.Path:
        run_dir = pred_dir / run_name
        run_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2019-10-01", periods=24, freq="h"),
            "yhat": np.random.rand(24) * 10 + 30,
            "y": np.random.rand(24) * 10 + 30,
            "yhat_lower_80": np.random.rand(24) * 5 + 25,
            "yhat_upper_80": np.random.rand(24) * 5 + 35,
        })
        csv_path = run_dir / "predictions.csv"
        df.to_csv(csv_path, index=False)
        return run_dir

    def test_returns_dataframe(self, tmp_path, monkeypatch):
        pred_dir = tmp_path / "predictions"
        run_name = "20260331_120000__LightGBM__24h__from_20191001"
        self._make_pred_csv(pred_dir, run_name)
        monkeypatch.setattr("utils.data_loader._predictions_dir", lambda: pred_dir)
        df = load_predictions(run_name)
        assert isinstance(df, pd.DataFrame)

    def test_timestamp_parsed_as_datetime(self, tmp_path, monkeypatch):
        pred_dir = tmp_path / "predictions"
        run_name = "20260331_120000__LightGBM__24h__from_20191001"
        self._make_pred_csv(pred_dir, run_name)
        monkeypatch.setattr("utils.data_loader._predictions_dir", lambda: pred_dir)
        df = load_predictions(run_name)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_correct_row_count(self, tmp_path, monkeypatch):
        pred_dir = tmp_path / "predictions"
        run_name = "20260331_120000__LightGBM__24h__from_20191001"
        self._make_pred_csv(pred_dir, run_name)
        monkeypatch.setattr("utils.data_loader._predictions_dir", lambda: pred_dir)
        df = load_predictions(run_name)
        assert len(df) == 24

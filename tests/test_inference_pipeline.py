"""Tests for InferencePipeline: filter string building, CI computation, and output structure."""

import json
import pathlib
import re
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from ev_load_fc.pipelines.inference_pipeline import InferencePipeline, InferencePipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path, model_family="XGBoost", feature_version="E_f_30_rfe_20", metric="rmse"):
    return InferencePipelineConfig(
        raw_hourly_path=tmp_path / "combined_processed.csv",
        feature_store=tmp_path,
        predictions_dir=tmp_path / "predictions",
        X_set="X_detrend",
        experiment_name="Test Experiment",
        model_family=model_family,
        feature_version=feature_version,
        metric=metric,
        horizon=24,
        inference_start=pd.Timestamp("2019-10-01 00:00:00"),
        confidence_intervals=[0.80, 0.95],
        n_bootstrap=10,
    )


def _make_dummy_fc_df(n=24):
    timestamps = pd.date_range("2019-10-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "yhat": np.random.rand(n) * 5 + 30,
        "y": np.random.rand(n) * 5 + 30,
    })


def _make_dummy_X(n=24):
    timestamps = pd.date_range("2019-10-01", periods=n, freq="h")
    return pd.DataFrame(
        np.random.rand(n, 5),
        index=timestamps,
        columns=[f"feat_{i}" for i in range(5)],
    )


# Lightweight stand-in classes so hasattr checks work without MagicMock auto-creation
class _MockRF:
    """Minimal RandomForest-like object with estimators_ attribute."""
    _FEATURE_COLS = [f"feat_{i}" for i in range(5)]

    def __init__(self, n_estimators=20, n_steps=24):
        self.feature_names_in_ = self._FEATURE_COLS
        self.estimators_ = []
        for _ in range(n_estimators):
            est = MagicMock()
            est.predict = MagicMock(return_value=np.random.rand(n_steps) * 5 + 30)
            self.estimators_.append(est)


class _MockUnknown:
    """Model with no estimators_ and an unrecognised type name."""
    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline(tmp_path):
    return InferencePipeline(config=_make_cfg(tmp_path))


# ---------------------------------------------------------------------------
# _build_filter_string
# ---------------------------------------------------------------------------

class TestBuildFilterString:

    def test_both_tags_produce_three_parts(self, pipeline):
        f = pipeline._build_filter_string()
        assert 'tags.level = "parent"' in f
        assert 'tags.model_family = "XGBoost"' in f
        assert 'tags.feature_set_version = "E_f_30_rfe_20"' in f
        assert f.count(" AND ") == 2

    def test_empty_model_family_excluded(self, tmp_path):
        p = InferencePipeline(config=_make_cfg(tmp_path, model_family=""))
        f = p._build_filter_string()
        assert "model_family" not in f
        assert 'tags.feature_set_version = "E_f_30_rfe_20"' in f

    def test_empty_feature_version_excluded(self, tmp_path):
        p = InferencePipeline(config=_make_cfg(tmp_path, feature_version=""))
        f = p._build_filter_string()
        assert "feature_set_version" not in f
        assert 'tags.model_family = "XGBoost"' in f

    def test_both_empty_leaves_only_level_filter(self, tmp_path):
        p = InferencePipeline(config=_make_cfg(tmp_path, model_family="", feature_version=""))
        f = p._build_filter_string()
        assert f == 'tags.level = "parent"'

    def test_parts_joined_with_and(self, pipeline):
        f = pipeline._build_filter_string()
        assert " AND " in f

    def test_catboost_model_family(self, tmp_path):
        p = InferencePipeline(config=_make_cfg(tmp_path, model_family="CatBoost"))
        f = p._build_filter_string()
        assert 'tags.model_family = "CatBoost"' in f

    def test_mae_metric_does_not_affect_filter_string(self, tmp_path):
        p = InferencePipeline(config=_make_cfg(tmp_path, metric="mae"))
        f = p._build_filter_string()
        # metric is used for ordering, not filtering — should not appear in filter string
        assert "metric" not in f


# ---------------------------------------------------------------------------
# CI computation — RandomForest path
# ---------------------------------------------------------------------------

class TestComputeCiRandomForest:

    def test_ci_columns_present(self, pipeline):
        pipeline.model = _MockRF()
        ci_df = pipeline._compute_ci(_make_dummy_X())
        assert set(ci_df.columns) == {"yhat_lower_80", "yhat_upper_80", "yhat_lower_95", "yhat_upper_95"}

    def test_lower_less_than_upper(self, pipeline):
        pipeline.model = _MockRF()
        ci_df = pipeline._compute_ci(_make_dummy_X())
        assert (ci_df["yhat_lower_80"] < ci_df["yhat_upper_80"]).all()
        assert (ci_df["yhat_lower_95"] < ci_df["yhat_upper_95"]).all()

    def test_95_wider_than_80(self, pipeline):
        pipeline.model = _MockRF()
        ci_df = pipeline._compute_ci(_make_dummy_X())
        width_80 = ci_df["yhat_upper_80"] - ci_df["yhat_lower_80"]
        width_95 = ci_df["yhat_upper_95"] - ci_df["yhat_lower_95"]
        assert (width_95 >= width_80).all()

    def test_symmetric_offsets_around_zero(self, pipeline):
        """CI offsets should be symmetric: lower = -upper (Gaussian assumption)."""
        pipeline.model = _MockRF()
        ci_df = pipeline._compute_ci(_make_dummy_X())
        np.testing.assert_allclose(
            ci_df["yhat_lower_80"].values,
            -ci_df["yhat_upper_80"].values,
            rtol=1e-10,
        )

    def test_single_ci_level(self, tmp_path):
        cfg = _make_cfg(tmp_path)
        cfg.confidence_intervals = [0.90]
        p = InferencePipeline(config=cfg)
        p.model = _MockRF()
        ci_df = p._compute_ci(_make_dummy_X())
        assert "yhat_lower_90" in ci_df.columns
        assert "yhat_upper_90" in ci_df.columns
        assert "yhat_lower_80" not in ci_df.columns

    def test_correct_row_count(self, pipeline):
        n = 48
        pipeline.model = _MockRF(n_steps=n)
        ci_df = pipeline._compute_ci(_make_dummy_X(n))
        assert len(ci_df) == n

    def test_uniform_estimators_give_zero_std(self, pipeline):
        """If all tree predictions are identical, std=0 → CI width=0."""
        model = _MockRF()
        constant = np.full(24, 5.0)
        for est in model.estimators_:
            est.predict = MagicMock(return_value=constant.copy())
        pipeline.model = model
        ci_df = pipeline._compute_ci(_make_dummy_X())
        np.testing.assert_allclose(ci_df["yhat_lower_80"].values, 0.0, atol=1e-12)
        np.testing.assert_allclose(ci_df["yhat_upper_80"].values, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# CI computation — unsupported model type
# ---------------------------------------------------------------------------

class TestComputeCiUnsupportedModel:

    def test_returns_empty_dataframe(self, pipeline):
        pipeline.model = _MockUnknown()
        pipeline.X = _make_dummy_X()  # needed for fallback feature extraction
        ci_df = pipeline._compute_ci(_make_dummy_X())
        assert isinstance(ci_df, pd.DataFrame)
        assert len(ci_df.columns) == 0


# ---------------------------------------------------------------------------
# _save_predictions
# ---------------------------------------------------------------------------

_DIR_PATTERN = re.compile(
    r"^\d{8}_\d{6}__\w+__\d+h__from_\d{8}$"
)


def _setup_save(pipeline, tmp_path, model_family="LightGBM"):
    """Helper: set pipeline state needed for _save_predictions."""
    pipeline._resolved_feature_version = "E_f_30_rfe_20"
    pipeline.cfg.predictions_dir = tmp_path / "predictions"
    pipeline.best_run = pd.Series({"tags.model_family": model_family})


class TestSavePredictions:

    def test_csv_written_inside_structured_dir(self, pipeline, tmp_path):
        _setup_save(pipeline, tmp_path)
        out_path = pipeline._save_predictions(_make_dummy_fc_df(), run_id="abc123")
        assert out_path.name == "predictions.csv"
        assert out_path.exists()
        assert _DIR_PATTERN.match(out_path.parent.name), (
            f"Dir name '{out_path.parent.name}' does not match expected pattern"
        )

    def test_dir_name_contains_model_family(self, pipeline, tmp_path):
        _setup_save(pipeline, tmp_path, model_family="CatBoost")
        out_path = pipeline._save_predictions(_make_dummy_fc_df(), run_id="abc123")
        assert "CatBoost" in out_path.parent.name

    def test_dir_name_contains_horizon(self, pipeline, tmp_path):
        _setup_save(pipeline, tmp_path)
        out_path = pipeline._save_predictions(_make_dummy_fc_df(), run_id="abc123")
        assert "24h" in out_path.parent.name

    def test_dir_name_contains_start_date(self, pipeline, tmp_path):
        _setup_save(pipeline, tmp_path)
        out_path = pipeline._save_predictions(_make_dummy_fc_df(), run_id="abc123")
        assert "from_20191001" in out_path.parent.name

    def test_metadata_json_written(self, pipeline, tmp_path):
        _setup_save(pipeline, tmp_path)
        out_path = pipeline._save_predictions(_make_dummy_fc_df(), run_id="abc123")
        meta_path = out_path.parent / "metadata.json"
        assert meta_path.exists()

    def test_metadata_json_schema(self, pipeline, tmp_path):
        _setup_save(pipeline, tmp_path)
        out_path = pipeline._save_predictions(_make_dummy_fc_df(), run_id="abc123")
        with open(out_path.parent / "metadata.json") as f:
            meta = json.load(f)
        required_keys = {"run_id", "model_family", "feature_version", "horizon",
                         "inference_start", "ci_levels", "metrics", "created_at"}
        assert required_keys == set(meta.keys())
        assert meta["run_id"] == "abc123"
        assert meta["model_family"] == "LightGBM"
        assert meta["horizon"] == 24
        assert isinstance(meta["ci_levels"], list)
        assert isinstance(meta["metrics"], dict)

    def test_metadata_metrics_computed_when_actuals_present(self, pipeline, tmp_path):
        _setup_save(pipeline, tmp_path)
        fc_df = _make_dummy_fc_df()  # has y column
        out_path = pipeline._save_predictions(fc_df, run_id="abc123")
        with open(out_path.parent / "metadata.json") as f:
            meta = json.load(f)
        assert "rmse" in meta["metrics"]
        assert "mae" in meta["metrics"]
        assert meta["metrics"]["rmse"] >= 0

    def test_csv_contains_all_columns(self, pipeline, tmp_path):
        _setup_save(pipeline, tmp_path)
        fc_df = _make_dummy_fc_df()
        out_path = pipeline._save_predictions(fc_df, run_id="run_xyz")
        loaded = pd.read_csv(out_path)
        assert list(loaded.columns) == list(fc_df.columns)
        assert len(loaded) == len(fc_df)

    def test_directory_created_if_missing(self, pipeline, tmp_path):
        _setup_save(pipeline, tmp_path)
        pipeline.cfg.predictions_dir = tmp_path / "new_dir" / "predictions"
        out_path = pipeline._save_predictions(_make_dummy_fc_df(), run_id="run1")
        assert out_path.exists()


# ---------------------------------------------------------------------------
# run() — integration smoke test with mocked external calls
# ---------------------------------------------------------------------------

class TestRunIntegration:

    def test_run_returns_dataframe_with_ci_columns(self, pipeline, tmp_path):
        n = 24
        fc_df = _make_dummy_fc_df(n)
        X_dummy = _make_dummy_X(n)

        mock_rf = _MockRF(n_steps=n)

        mock_run = pd.Series({
            "run_id": "test_run_id_123",
            "tags.feature_set_version": "E_f_30_rfe_20",
            "tags.model_family": "RandomForest",
        })

        pipeline.cfg.predictions_dir = tmp_path / "predictions"
        pipeline.cfg.feature_version = "E_f_30_rfe_20"

        def _fake_load_model():
            pipeline.model = mock_rf
            pipeline.best_run = mock_run

        def _fake_load_data():
            pipeline.raw_hourly = pd.DataFrame(
                {"energy": np.ones(20000)},
                index=pd.date_range("2018-01-01", periods=20000, freq="h"),
            )
            pipeline.X = X_dummy
            pipeline._resolved_feature_version = "E_f_30_rfe_20"

        with (
            patch("ev_load_fc.pipelines.inference_pipeline.init_mlflow"),
            patch.object(pipeline, "_load_model", side_effect=_fake_load_model),
            patch.object(pipeline, "_load_data", side_effect=_fake_load_data),
            patch("ev_load_fc.pipelines.inference_pipeline.recursive_forecast", return_value=fc_df),
        ):
            result = pipeline.run()

        assert isinstance(result, pd.DataFrame)
        for col in ["timestamp", "yhat", "y", "yhat_lower_80", "yhat_upper_80", "yhat_lower_95", "yhat_upper_95"]:
            assert col in result.columns, f"Missing column: {col}"
        assert len(result) == n

    def test_run_saves_csv_to_disk(self, pipeline, tmp_path):
        n = 24
        fc_df = _make_dummy_fc_df(n)
        X_dummy = _make_dummy_X(n)
        mock_rf = _MockRF(n_steps=n)
        mock_run = pd.Series({
            "run_id": "saved_run_abc",
            "tags.feature_set_version": "E_f_30_rfe_20",
            "tags.model_family": "RandomForest",
        })

        pipeline.cfg.predictions_dir = tmp_path / "predictions"
        pipeline.cfg.feature_version = "E_f_30_rfe_20"

        def _fake_load_model():
            pipeline.model = mock_rf
            pipeline.best_run = mock_run

        def _fake_load_data():
            pipeline.raw_hourly = pd.DataFrame(
                {"energy": np.ones(20000)},
                index=pd.date_range("2018-01-01", periods=20000, freq="h"),
            )
            pipeline.X = X_dummy
            pipeline._resolved_feature_version = "E_f_30_rfe_20"

        with (
            patch("ev_load_fc.pipelines.inference_pipeline.init_mlflow"),
            patch.object(pipeline, "_load_model", side_effect=_fake_load_model),
            patch.object(pipeline, "_load_data", side_effect=_fake_load_data),
            patch("ev_load_fc.pipelines.inference_pipeline.recursive_forecast", return_value=fc_df),
        ):
            pipeline.run()

        predictions_dir = tmp_path / "predictions"
        run_dirs = list(predictions_dir.iterdir())
        assert len(run_dirs) == 1, "Expected exactly one inference run directory"
        assert (run_dirs[0] / "predictions.csv").exists()
        assert (run_dirs[0] / "metadata.json").exists()


# ---------------------------------------------------------------------------
# CI fix — verify lower < yhat < upper in full run() output
# ---------------------------------------------------------------------------

class TestCIFixInRunOutput:
    """Verify the CI calculation bug fix: lower bounds < yhat, upper bounds > yhat."""

    def _run_pipeline_with_mock_rf(self, pipeline, tmp_path, n=24):
        fc_df = _make_dummy_fc_df(n)
        X_dummy = _make_dummy_X(n)
        mock_rf = _MockRF(n_steps=n)
        mock_run = pd.Series({
            "run_id": "ci_fix_run",
            "tags.feature_set_version": "E_f_30_rfe_20",
            "tags.model_family": "RandomForest",
        })

        pipeline.cfg.predictions_dir = tmp_path / "predictions"
        pipeline.cfg.feature_version = "E_f_30_rfe_20"

        def _fake_load_model():
            pipeline.model = mock_rf
            pipeline.best_run = mock_run

        def _fake_load_data():
            pipeline.raw_hourly = pd.DataFrame(
                {"energy": np.ones(20000)},
                index=pd.date_range("2018-01-01", periods=20000, freq="h"),
            )
            pipeline.X = X_dummy
            pipeline._resolved_feature_version = "E_f_30_rfe_20"

        with (
            patch("ev_load_fc.pipelines.inference_pipeline.init_mlflow"),
            patch.object(pipeline, "_load_model", side_effect=_fake_load_model),
            patch.object(pipeline, "_load_data", side_effect=_fake_load_data),
            patch("ev_load_fc.pipelines.inference_pipeline.recursive_forecast", return_value=fc_df),
        ):
            return pipeline.run()

    def test_lower_80_less_than_yhat(self, pipeline, tmp_path):
        result = self._run_pipeline_with_mock_rf(pipeline, tmp_path)
        assert (result["yhat_lower_80"] <= result["yhat"]).all(), \
            "lower_80 must be <= yhat for all steps"

    def test_upper_80_greater_than_yhat(self, pipeline, tmp_path):
        result = self._run_pipeline_with_mock_rf(pipeline, tmp_path)
        assert (result["yhat_upper_80"] >= result["yhat"]).all(), \
            "upper_80 must be >= yhat for all steps"

    def test_lower_95_less_than_lower_80(self, pipeline, tmp_path):
        result = self._run_pipeline_with_mock_rf(pipeline, tmp_path)
        assert (result["yhat_lower_95"] <= result["yhat_lower_80"]).all(), \
            "95% CI lower must be <= 80% CI lower (wider interval)"

    def test_upper_95_greater_than_upper_80(self, pipeline, tmp_path):
        result = self._run_pipeline_with_mock_rf(pipeline, tmp_path)
        assert (result["yhat_upper_95"] >= result["yhat_upper_80"]).all(), \
            "95% CI upper must be >= 80% CI upper (wider interval)"
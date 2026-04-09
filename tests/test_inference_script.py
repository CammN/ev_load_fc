"""Smoke tests for scripts/inference.py (InferencePipeline entry point).

Covers the script-level wiring: build_pipeline_params(), CLI overrides, and main().
Pipeline-class internals are tested separately in test_inference_pipeline.py.
"""
import pathlib
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import inference
from ev_load_fc.config import CFG


# ---------------------------------------------------------------------------
# build_pipeline_params
# ---------------------------------------------------------------------------

class TestBuildPipelineParams:

    def test_defaults_use_config_horizon(self):
        params = inference.build_pipeline_params()
        assert params["horizon"] == CFG["inference"]["horizon"]

    def test_defaults_use_config_metric(self):
        params = inference.build_pipeline_params()
        assert params["metric"] == CFG["inference"]["metric"]

    def test_defaults_use_config_experiment_name(self):
        params = inference.build_pipeline_params()
        assert params["experiment_name"] == CFG["inference"]["experiment_name"]

    def test_horizon_override(self):
        params = inference.build_pipeline_params(horizon=48)
        assert params["horizon"] == 48

    def test_metric_override_mae(self):
        params = inference.build_pipeline_params(metric="mae")
        assert params["metric"] == "mae"

    def test_metric_override_rmse(self):
        params = inference.build_pipeline_params(metric="rmse")
        assert params["metric"] == "rmse"

    def test_inference_start_override(self):
        params = inference.build_pipeline_params(inference_start="2020-01-01 00:00:00")
        assert params["inference_start"] == pd.Timestamp("2020-01-01 00:00:00")

    def test_inference_start_is_pd_Timestamp(self):
        params = inference.build_pipeline_params()
        assert isinstance(params["inference_start"], pd.Timestamp)

    def test_model_family_override(self):
        params = inference.build_pipeline_params(model_family="CatBoost")
        assert params["model_family"] == "CatBoost"

    def test_feature_version_override(self):
        params = inference.build_pipeline_params(feature_version="E_f_30_rfe_20")
        assert params["feature_version"] == "E_f_30_rfe_20"

    def test_empty_model_family_uses_config(self):
        params = inference.build_pipeline_params()
        # None passed → uses config value (empty string or set value)
        assert params["model_family"] == CFG["inference"].get("model_family", "")

    def test_path_fields_are_pathlib_Path(self):
        params = inference.build_pipeline_params()
        assert isinstance(params["raw_hourly_path"], pathlib.Path)
        assert isinstance(params["feature_store"], pathlib.Path)
        assert isinstance(params["predictions_dir"], pathlib.Path)

    def test_confidence_intervals_is_list(self):
        params = inference.build_pipeline_params()
        assert isinstance(params["confidence_intervals"], list)
        assert len(params["confidence_intervals"]) > 0

    def test_confidence_intervals_values_between_0_and_1(self):
        params = inference.build_pipeline_params()
        for ci in params["confidence_intervals"]:
            assert 0 < ci < 1, f"CI level {ci} not in (0, 1)"

    def test_n_bootstrap_not_in_params(self):
        params = inference.build_pipeline_params()
        assert "n_bootstrap" not in params


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _make_dummy_fc_df(n=24):
    timestamps = pd.date_range("2019-10-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "yhat": np.random.rand(n) * 5 + 30,
        "y": np.random.rand(n) * 5 + 30,
    })


def _run_main_capture_config(argv, fc_df=None):
    """Run main() with given sys.argv; capture the InferencePipelineConfig and return it."""
    if fc_df is None:
        fc_df = _make_dummy_fc_df()

    captured = {}

    def fake_pipeline(config):
        captured["config"] = config
        mock = MagicMock()
        mock.run.return_value = fc_df
        return mock

    with (
        patch("sys.argv", argv),
        patch.object(inference, "setup_logging", return_value=MagicMock()),
        patch.object(inference, "InferencePipeline", side_effect=fake_pipeline),
    ):
        inference.main()

    return captured["config"]


class TestMain:

    def test_no_cli_args_uses_config_defaults(self):
        cfg = _run_main_capture_config(["inference.py"])
        assert cfg.horizon == CFG["inference"]["horizon"]
        assert cfg.metric == CFG["inference"]["metric"]

    def test_horizon_cli_override(self):
        cfg = _run_main_capture_config(["inference.py", "--horizon", "48"])
        assert cfg.horizon == 48

    def test_metric_cli_override_mae(self):
        cfg = _run_main_capture_config(["inference.py", "--metric", "mae"])
        assert cfg.metric == "mae"

    def test_inference_start_cli_override(self):
        cfg = _run_main_capture_config(["inference.py", "--inference_start", "2020-06-01 00:00:00"])
        assert cfg.inference_start == pd.Timestamp("2020-06-01 00:00:00")

    def test_model_family_cli_override(self):
        cfg = _run_main_capture_config(["inference.py", "--model_family", "XGBoost"])
        assert cfg.model_family == "XGBoost"

    def test_feature_version_cli_override(self):
        cfg = _run_main_capture_config(["inference.py", "--feature_version", "E_f_30_rfe_20"])
        assert cfg.feature_version == "E_f_30_rfe_20"

    def test_pipeline_run_called_once(self):
        mock_instance = MagicMock()
        mock_instance.run.return_value = _make_dummy_fc_df()

        with (
            patch("sys.argv", ["inference.py"]),
            patch.object(inference, "setup_logging", return_value=MagicMock()),
            patch.object(inference, "InferencePipeline", return_value=mock_instance),
        ):
            inference.main()

        mock_instance.run.assert_called_once()

    def test_main_returns_dataframe(self):
        fc_df = _make_dummy_fc_df()
        mock_instance = MagicMock()
        mock_instance.run.return_value = fc_df

        with (
            patch("sys.argv", ["inference.py"]),
            patch.object(inference, "setup_logging", return_value=MagicMock()),
            patch.object(inference, "InferencePipeline", return_value=mock_instance),
        ):
            result = inference.main()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(fc_df)

    def test_config_path_fields_are_pathlib_Path(self):
        cfg = _run_main_capture_config(["inference.py"])
        assert isinstance(cfg.raw_hourly_path, pathlib.Path)
        assert isinstance(cfg.predictions_dir, pathlib.Path)

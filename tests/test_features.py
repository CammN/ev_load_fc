"""Smoke tests for scripts/features.py (FeaturePipeline entry point)."""
import pathlib
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import features
from ev_load_fc.config import CFG


# ---------------------------------------------------------------------------
# build_pipeline_params
# ---------------------------------------------------------------------------

class TestBuildPipelineParams:

    def test_both_flags_true_sets_run_fields(self):
        params = features.build_pipeline_params(fe=True, fs=True)
        assert params["run_feat_eng"] is True
        assert params["run_feat_select"] is True

    def test_fe_only(self):
        params = features.build_pipeline_params(fe=True, fs=False)
        assert params["run_feat_eng"] is True
        assert params["run_feat_select"] is False

    def test_fs_only(self):
        params = features.build_pipeline_params(fe=False, fs=True)
        assert params["run_feat_eng"] is False
        assert params["run_feat_select"] is True

    def test_path_fields_are_pathlib_Path(self):
        params = features.build_pipeline_params(fe=True, fs=True)
        assert isinstance(params["combined_path"], pathlib.Path)
        assert isinstance(params["feature_store"], pathlib.Path)

    def test_timestamp_fields_are_pd_Timestamp(self):
        params = features.build_pipeline_params(fe=True, fs=True)
        assert isinstance(params["min_timestamp"], pd.Timestamp)
        assert isinstance(params["max_timestamp"], pd.Timestamp)
        assert isinstance(params["split_date"], pd.Timestamp)

    def test_holidays_is_set(self):
        params = features.build_pipeline_params(fe=True, fs=True)
        assert isinstance(params["holidays"], set)

    def test_holiday_list_is_list(self):
        params = features.build_pipeline_params(fe=True, fs=True)
        assert isinstance(params["holiday_list"], list)

    def test_holiday_list_and_holidays_same_elements(self):
        params = features.build_pipeline_params(fe=True, fs=True)
        assert set(params["holiday_list"]) == params["holidays"]

    def test_time_feature_dict_is_dict(self):
        params = features.build_pipeline_params(fe=True, fs=True)
        assert isinstance(params["time_feature_dict"], dict)

    def test_column_substr_fields_are_lists(self):
        params = features.build_pipeline_params(fe=True, fs=True)
        for key in ("energy_col_substrs", "weather_col_substrs",
                    "temperature_col_substrs", "traffic_col_substrs"):
            assert isinstance(params[key], list), f"{key} is not a list"

    def test_k1_k2_are_integers(self):
        params = features.build_pipeline_params(fe=True, fs=True)
        assert isinstance(params["k_1"], int)
        assert isinstance(params["k_2"], int)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _run_main_capture_config(argv):
    captured = {}

    def fake_pipeline(config):
        captured["config"] = config
        return MagicMock()

    with (
        patch("sys.argv", argv),
        patch.object(features, "setup_logging", return_value=MagicMock()),
        patch.object(features, "FeaturePipeline", side_effect=fake_pipeline),
    ):
        features.main()

    return captured["config"]


class TestMain:

    def test_no_cli_flags_runs_both_steps(self):
        cfg = _run_main_capture_config(["features.py"])
        assert cfg.run_feat_eng is True
        assert cfg.run_feat_select is True

    def test_fe_flag_only(self):
        cfg = _run_main_capture_config(["features.py", "--fe"])
        assert cfg.run_feat_eng is True
        assert cfg.run_feat_select is False

    def test_fs_flag_only(self):
        cfg = _run_main_capture_config(["features.py", "--fs"])
        assert cfg.run_feat_eng is False
        assert cfg.run_feat_select is True

    def test_pipeline_run_called_once(self):
        mock_instance = MagicMock()

        with (
            patch("sys.argv", ["features.py"]),
            patch.object(features, "setup_logging", return_value=MagicMock()),
            patch.object(features, "FeaturePipeline", return_value=mock_instance),
        ):
            features.main()

        mock_instance.run.assert_called_once()

    def test_config_path_fields_are_pathlib_Path(self):
        cfg = _run_main_capture_config(["features.py"])
        assert isinstance(cfg.combined_path, pathlib.Path)
        assert isinstance(cfg.feature_store, pathlib.Path)

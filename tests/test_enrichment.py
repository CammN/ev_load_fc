"""Smoke tests for scripts/enrichment.py (PreprocessingPipeline entry point)."""
import pathlib
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import enrichment
from ev_load_fc.config import CFG


# ---------------------------------------------------------------------------
# build_pipeline_params
# ---------------------------------------------------------------------------

class TestBuildPipelineParams:

    def test_all_flags_true_sets_all_run_fields(self):
        params = enrichment.build_pipeline_params(
            ev=True, weather=True, temperature=True, traffic=True, combine=True
        )
        assert params["run_ev"] is True
        assert params["run_weather"] is True
        assert params["run_temperature"] is True
        assert params["run_traffic"] is True
        assert params["run_combine"] is True

    def test_ev_only_clears_other_flags(self):
        params = enrichment.build_pipeline_params(
            ev=True, weather=False, temperature=False, traffic=False, combine=False
        )
        assert params["run_ev"] is True
        assert params["run_weather"] is False
        assert params["run_temperature"] is False
        assert params["run_traffic"] is False
        assert params["run_combine"] is False

    def test_combine_only(self):
        params = enrichment.build_pipeline_params(
            ev=False, weather=False, temperature=False, traffic=False, combine=True
        )
        assert params["run_combine"] is True
        assert params["run_ev"] is False

    def test_path_fields_are_pathlib_Path(self):
        params = enrichment.build_pipeline_params(
            ev=True, weather=True, temperature=True, traffic=True, combine=True
        )
        for key in ("processed_data_path", "ev_int_path", "weather_int_path", "temp_path",
                    "traffic_int_path", "ev_proc_path", "weather_proc_path",
                    "temp_proc_path", "traffic_proc_path", "combined_path"):
            assert isinstance(params[key], pathlib.Path), f"{key} is not a pathlib.Path"

    def test_timestamp_fields_are_pd_Timestamp(self):
        params = enrichment.build_pipeline_params(
            ev=True, weather=True, temperature=True, traffic=True, combine=True
        )
        assert isinstance(params["min_timestamp"], pd.Timestamp)
        assert isinstance(params["max_timestamp"], pd.Timestamp)
        assert isinstance(params["split_date"], pd.Timestamp)

    def test_split_date_between_min_and_max(self):
        params = enrichment.build_pipeline_params(
            ev=True, weather=True, temperature=True, traffic=True, combine=True
        )
        assert params["min_timestamp"] <= params["split_date"] <= params["max_timestamp"]

    def test_column_filter_fields_are_lists(self):
        params = enrichment.build_pipeline_params(
            ev=True, weather=True, temperature=True, traffic=True, combine=True
        )
        assert isinstance(params["ev_cols"], list)
        assert isinstance(params["weather_cols"], list)
        assert isinstance(params["traffic_cols"], list)


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
        patch.object(enrichment, "setup_logging", return_value=MagicMock()),
        patch.object(enrichment, "PreprocessingPipeline", side_effect=fake_pipeline),
    ):
        enrichment.main()

    return captured["config"]


class TestMain:

    def test_no_cli_flags_runs_all_steps(self):
        cfg = _run_main_capture_config(["enrichment.py"])
        assert cfg.run_ev is True
        assert cfg.run_weather is True
        assert cfg.run_temperature is True
        assert cfg.run_traffic is True
        assert cfg.run_combine is True

    def test_ev_flag_only_runs_ev(self):
        cfg = _run_main_capture_config(["enrichment.py", "--ev"])
        assert cfg.run_ev is True
        assert cfg.run_weather is False
        assert cfg.run_temperature is False
        assert cfg.run_traffic is False
        assert cfg.run_combine is False

    def test_combine_flag_only_runs_combine(self):
        cfg = _run_main_capture_config(["enrichment.py", "--combine"])
        assert cfg.run_combine is True
        assert cfg.run_ev is False
        assert cfg.run_weather is False

    def test_weather_and_temperature_flags(self):
        cfg = _run_main_capture_config(["enrichment.py", "--weather", "--temperature"])
        assert cfg.run_weather is True
        assert cfg.run_temperature is True
        assert cfg.run_ev is False
        assert cfg.run_combine is False

    def test_pipeline_run_called_once(self):
        mock_instance = MagicMock()

        with (
            patch("sys.argv", ["enrichment.py"]),
            patch.object(enrichment, "setup_logging", return_value=MagicMock()),
            patch.object(enrichment, "PreprocessingPipeline", return_value=mock_instance),
        ):
            enrichment.main()

        mock_instance.run.assert_called_once()

    def test_config_path_fields_are_pathlib_Path(self):
        cfg = _run_main_capture_config(["enrichment.py"])
        assert isinstance(cfg.processed_data_path, pathlib.Path)
        assert isinstance(cfg.combined_path, pathlib.Path)

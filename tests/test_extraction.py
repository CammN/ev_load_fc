"""Smoke tests for scripts/extraction.py (LoadingPipeline entry point)."""
import pathlib
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import extraction
from ev_load_fc.config import CFG


# ---------------------------------------------------------------------------
# build_pipeline_params
# ---------------------------------------------------------------------------

class TestBuildPipelineParams:

    def test_all_flags_true_sets_run_fields(self):
        params = extraction.build_pipeline_params(ev=True, weather=True, traffic=True)
        assert params["run_ev"] is True
        assert params["run_weather"] is True
        assert params["run_traffic"] is True

    def test_ev_only_clears_other_flags(self):
        params = extraction.build_pipeline_params(ev=True, weather=False, traffic=False)
        assert params["run_ev"] is True
        assert params["run_weather"] is False
        assert params["run_traffic"] is False

    def test_weather_only_clears_other_flags(self):
        params = extraction.build_pipeline_params(ev=False, weather=True, traffic=False)
        assert params["run_ev"] is False
        assert params["run_weather"] is True
        assert params["run_traffic"] is False

    def test_path_fields_are_pathlib_Path(self):
        params = extraction.build_pipeline_params(ev=True, weather=True, traffic=True)
        for key in ("ev_raw_path", "weather_raw_path", "traffic_raw_path",
                    "ev_int_path", "weather_int_path", "traffic_int_path", "temp_path"):
            assert isinstance(params[key], pathlib.Path), f"{key} is not a pathlib.Path"

    def test_timestamp_fields_are_pd_Timestamp(self):
        params = extraction.build_pipeline_params(ev=True, weather=True, traffic=True)
        assert isinstance(params["min_timestamp"], pd.Timestamp)
        assert isinstance(params["max_timestamp"], pd.Timestamp)

    def test_min_timestamp_before_max(self):
        params = extraction.build_pipeline_params(ev=True, weather=True, traffic=True)
        assert params["min_timestamp"] < params["max_timestamp"]

    def test_city_filters_are_lists(self):
        params = extraction.build_pipeline_params(ev=True, weather=True, traffic=True)
        assert isinstance(params["weather_cities"], list)
        assert isinstance(params["traffic_cities"], list)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _run_main_capture_config(argv):
    """Run main() with given sys.argv and return the LoadingPipelineConfig passed to the pipeline."""
    captured = {}

    def fake_pipeline(config):
        captured["config"] = config
        return MagicMock()

    with (
        patch("sys.argv", argv),
        patch.object(extraction, "setup_logging", return_value=MagicMock()),
        patch.object(extraction, "LoadingPipeline", side_effect=fake_pipeline),
    ):
        extraction.main()

    return captured["config"]


class TestMain:

    def test_no_cli_flags_runs_all_steps(self):
        cfg = _run_main_capture_config(["extraction.py"])
        assert cfg.run_ev is True
        assert cfg.run_weather is True
        assert cfg.run_traffic is True

    def test_ev_flag_only_runs_ev(self):
        cfg = _run_main_capture_config(["extraction.py", "--ev"])
        assert cfg.run_ev is True
        assert cfg.run_weather is False
        assert cfg.run_traffic is False

    def test_weather_flag_only_runs_weather(self):
        cfg = _run_main_capture_config(["extraction.py", "--weather"])
        assert cfg.run_ev is False
        assert cfg.run_weather is True
        assert cfg.run_traffic is False

    def test_traffic_flag_only_runs_traffic(self):
        cfg = _run_main_capture_config(["extraction.py", "--traffic"])
        assert cfg.run_ev is False
        assert cfg.run_weather is False
        assert cfg.run_traffic is True

    def test_pipeline_run_called_once(self):
        mock_instance = MagicMock()

        with (
            patch("sys.argv", ["extraction.py"]),
            patch.object(extraction, "setup_logging", return_value=MagicMock()),
            patch.object(extraction, "LoadingPipeline", return_value=mock_instance),
        ):
            extraction.main()

        mock_instance.run.assert_called_once()

    def test_config_path_fields_are_pathlib_Path(self):
        cfg = _run_main_capture_config(["extraction.py"])
        assert isinstance(cfg.ev_raw_path, pathlib.Path)
        assert isinstance(cfg.ev_int_path, pathlib.Path)
        assert isinstance(cfg.temp_path, pathlib.Path)

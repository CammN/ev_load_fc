"""Smoke tests for scripts/training.py (TrainingPipeline entry point)."""
import pathlib
import pytest
from unittest.mock import MagicMock, patch

import training
from ev_load_fc.config import CFG


# ---------------------------------------------------------------------------
# build_pipeline_params
# ---------------------------------------------------------------------------

class TestBuildPipelineParams:

    def test_defaults_use_config_experiment_name(self):
        params = training.build_pipeline_params()
        assert params["experiment_name"] == CFG["training"]["mlflow"]["experiment_name"]

    def test_defaults_use_config_models_to_run(self):
        params = training.build_pipeline_params()
        assert params["models_to_run"] == CFG["training"]["optuna"]["models_to_run"]

    def test_exp_name_override(self):
        # argparse nargs="+" passes a list; build_pipeline_params stores it as-is
        params = training.build_pipeline_params(exp_name=["Custom Experiment"])
        assert params["experiment_name"] == ["Custom Experiment"]

    def test_model_names_override_single(self):
        params = training.build_pipeline_params(model_names=["XGBoost"])
        assert params["models_to_run"] == ["XGBoost"]

    def test_model_names_override_multiple(self):
        params = training.build_pipeline_params(model_names=["XGBoost", "CatBoost"])
        assert params["models_to_run"] == ["XGBoost", "CatBoost"]

    def test_path_fields_are_pathlib_Path(self):
        params = training.build_pipeline_params()
        for key in ("feature_store", "configs", "images"):
            assert isinstance(params[key], pathlib.Path), f"{key} is not a pathlib.Path"

    def test_trials_is_integer(self):
        params = training.build_pipeline_params()
        assert isinstance(params["trials"], int)

    def test_splits_is_integer(self):
        params = training.build_pipeline_params()
        assert isinstance(params["splits"], int)

    def test_metric_is_string(self):
        params = training.build_pipeline_params()
        assert isinstance(params["metric"], str)
        assert params["metric"] in ("rmse", "mae")

    def test_search_spaces_is_dict(self):
        params = training.build_pipeline_params()
        assert isinstance(params["search_spaces"], dict)

    def test_seed_is_integer(self):
        params = training.build_pipeline_params()
        assert isinstance(params["seed"], int)


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
        patch.object(training, "setup_logging", return_value=MagicMock()),
        patch.object(training, "TrainingPipeline", side_effect=fake_pipeline),
    ):
        training.main()

    return captured["config"]


class TestMain:

    def test_no_cli_args_uses_config_defaults(self):
        cfg = _run_main_capture_config(["training.py"])
        assert cfg.experiment_name == CFG["training"]["mlflow"]["experiment_name"]
        assert cfg.models_to_run == CFG["training"]["optuna"]["models_to_run"]

    def test_model_names_cli_override(self):
        cfg = _run_main_capture_config(["training.py", "--model_names", "XGBoost"])
        assert cfg.models_to_run == ["XGBoost"]

    def test_exp_name_cli_override(self):
        # nargs="+" on --exp_name means argparse produces a list, stored as-is
        cfg = _run_main_capture_config(["training.py", "--exp_name", "My Test Experiment"])
        assert cfg.experiment_name == ["My Test Experiment"]

    def test_pipeline_run_called_once(self):
        mock_instance = MagicMock()

        with (
            patch("sys.argv", ["training.py"]),
            patch.object(training, "setup_logging", return_value=MagicMock()),
            patch.object(training, "TrainingPipeline", return_value=mock_instance),
        ):
            training.main()

        mock_instance.run.assert_called_once()

    def test_config_path_fields_are_pathlib_Path(self):
        cfg = _run_main_capture_config(["training.py"])
        assert isinstance(cfg.feature_store, pathlib.Path)
        assert isinstance(cfg.configs, pathlib.Path)
        assert isinstance(cfg.images, pathlib.Path)

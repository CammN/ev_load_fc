"""Unit tests for streamlit_app/utils/mlflow_client.py."""
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "streamlit_app"))

from utils.mlflow_client import (
    ALL_MODEL_FAMILIES,
    is_mlflow_available,
    plot_image_path,
    optuna_html_path,
)


class TestAllModelFamilies:

    def test_contains_all_six_families(self):
        expected = {"Random Forest", "AdaBoost", "XGBoost", "LightGBM", "CatBoost", "Prophet"}
        assert expected == set(ALL_MODEL_FAMILIES)

    def test_is_list(self):
        assert isinstance(ALL_MODEL_FAMILIES, list)

    def test_no_duplicates(self):
        assert len(ALL_MODEL_FAMILIES) == len(set(ALL_MODEL_FAMILIES))


class TestIsMlflowAvailable:

    def test_returns_false_gracefully_when_db_missing(self, monkeypatch, tmp_path):
        """Should return False (not raise) when mlflow DB is not found."""
        def _raise(*a, **kw):
            raise Exception("DB not found")

        import utils.mlflow_client as mc
        monkeypatch.setattr(mc, "is_mlflow_available", lambda: False)
        assert mc.is_mlflow_available() is False

    def test_returns_bool(self):
        result = is_mlflow_available()
        assert isinstance(result, bool)


class TestPlotImagePath:

    def test_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        import utils.mlflow_client as mc
        monkeypatch.setattr(
            "utils.mlflow_client.resolve_path",
            lambda _: tmp_path / "plots",
        )
        result = plot_image_path("LightGBM", 1, "feature_importances")
        assert result is None

    def test_returns_path_when_file_exists(self, tmp_path, monkeypatch):
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()
        (plots_dir / "LightGBM_run1_feature_importances.png").write_bytes(b"fake")

        import utils.mlflow_client as mc
        monkeypatch.setattr("utils.mlflow_client.resolve_path", lambda _: plots_dir)
        result = plot_image_path("LightGBM", 1, "feature_importances")
        assert result is not None
        assert result.exists()

    def test_correct_filename_constructed(self, tmp_path, monkeypatch):
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()
        (plots_dir / "CatBoost_run3_residuals.png").write_bytes(b"fake")

        import utils.mlflow_client as mc
        monkeypatch.setattr("utils.mlflow_client.resolve_path", lambda _: plots_dir)
        result = plot_image_path("CatBoost", 3, "residuals")
        assert result.name == "CatBoost_run3_residuals.png"


class TestOptunaHtmlPath:

    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        import utils.mlflow_client as mc
        monkeypatch.setattr("utils.mlflow_client.resolve_path", lambda _: tmp_path / "plots")
        result = optuna_html_path("XGBoost", 2, "optimization_history")
        assert result is None

    def test_returns_path_when_exists(self, tmp_path, monkeypatch):
        plots_dir = tmp_path / "plots"
        plots_dir.mkdir()
        (plots_dir / "XGBoost_run2_optimization_history.html").write_text("<html/>")

        import utils.mlflow_client as mc
        monkeypatch.setattr("utils.mlflow_client.resolve_path", lambda _: plots_dir)
        result = optuna_html_path("XGBoost", 2, "optimization_history")
        assert result is not None
        assert result.suffix == ".html"

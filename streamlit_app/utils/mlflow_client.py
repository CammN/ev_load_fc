"""MLflow query helpers for the Streamlit app.

All functions gracefully return empty results (not exceptions) when MLflow is unavailable,
so the app degrades gracefully on Streamlit Cloud without a local mlflow.db.
"""
import pathlib
import pandas as pd

from ev_load_fc.config import resolve_path

# Canonical list of all supported model families
ALL_MODEL_FAMILIES = [
    "Random Forest",
    "AdaBoost",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Prophet",
]

# Map display name → MLflow tag value
_FAMILY_TAG = {
    "Random Forest": "RandomForest",
    "AdaBoost": "AdaBoost",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
    "Prophet": "Prophet",
}


def is_mlflow_available() -> bool:
    """Return True if the local MLflow DB can be reached."""
    try:
        from ev_load_fc.training.mlflow_api import init_mlflow
        import mlflow
        init_mlflow()
        mlflow.search_experiments()
        return True
    except Exception:
        return False


def get_experiment_names() -> list[str]:
    """Return all MLflow experiment names (excluding the default experiment 0)."""
    try:
        from ev_load_fc.training.mlflow_api import init_mlflow
        import mlflow
        init_mlflow()
        experiments = mlflow.search_experiments()
        return [e.name for e in experiments if e.name != "Default"]
    except Exception:
        return []


def get_runs_summary(experiment_name: str) -> pd.DataFrame:
    """Return a summary DataFrame of all parent runs in the given experiment.

    Columns: model_family, rmse, mae, n_trials, feature_version, run_id
    """
    try:
        from ev_load_fc.training.mlflow_api import init_mlflow
        import mlflow
        init_mlflow()
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string='tags.level = "parent" AND status = "FINISHED"',
            order_by=["metrics.best_rmse ASC"],
        )
        if runs.empty:
            return pd.DataFrame()

        summary = pd.DataFrame({
            "model_family": runs.get("tags.model_family", pd.Series(dtype=str)),
            "rmse": runs.get("metrics.best_rmse", pd.Series(dtype=float)).round(4),
            "mae": runs.get("metrics.best_mae", pd.Series(dtype=float)).round(4),
            "feature_version": runs.get("tags.feature_set_version", pd.Series(dtype=str)),
            "run_id": runs["run_id"].str[:8],  # truncate for display
            "run_id_full": runs["run_id"],
        })
        return summary.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def get_best_run_per_family(experiment_name: str) -> pd.DataFrame:
    """Return the best run (by RMSE) for each model family that has runs.

    Columns: model_family, rmse, mae, feature_version, run_id_full
    """
    summary = get_runs_summary(experiment_name)
    if summary.empty:
        return pd.DataFrame()

    return (
        summary.sort_values("rmse")
        .drop_duplicates(subset=["model_family"], keep="first")
        .reset_index(drop=True)
    )


def get_available_families(experiment_name: str) -> list[str]:
    """Return display names of model families that have at least one MLflow run."""
    best = get_best_run_per_family(experiment_name)
    if best.empty:
        return []
    tag_to_display = {v: k for k, v in _FAMILY_TAG.items()}
    return [tag_to_display.get(f, f) for f in best["model_family"].tolist()]


def plot_image_path(model_family: str, run_num: int, plot_type: str) -> pathlib.Path | None:
    """Return the path to a saved evaluation plot image, or None if it doesn't exist.

    Args:
        model_family: MLflow tag value e.g. "LightGBM"
        run_num: Run number suffix used in the filename e.g. 1
        plot_type: One of "correlations", "feature_importances", "residuals"

    Returns:
        Absolute Path if the file exists, else None.
    """
    plots_dir = resolve_path("images/plots")
    filename = f"{model_family}_run{run_num}_{plot_type}.png"
    path = plots_dir / filename
    return path if path.exists() else None


def optuna_html_path(model_family: str, run_num: int, plot_type: str) -> pathlib.Path | None:
    """Return path to an Optuna HTML plot, or None if it doesn't exist.

    Args:
        plot_type: "optimization_history" or "param_importances"
    """
    plots_dir = resolve_path("images/plots")
    filename = f"{model_family}_run{run_num}_{plot_type}.html"
    path = plots_dir / filename
    return path if path.exists() else None

"""Cached data loaders for the EV Load Forecasting Streamlit app.

All paths are resolved relative to PROJECT_ROOT from the project's config module,
so the app works regardless of the working directory it's launched from.
"""
import json
import pathlib
import pandas as pd
import streamlit as st

from ev_load_fc.config import resolve_path, CFG


def _predictions_dir() -> pathlib.Path:
    return resolve_path(CFG["paths"]["predictions"])


@st.cache_data
def load_processed() -> pd.DataFrame:
    """Load combined_processed.csv (hourly energy + weather + traffic)."""
    path = resolve_path(CFG["paths"]["processed_data"]) / CFG["files"]["combined_filename"]
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


@st.cache_data
def load_train_features() -> pd.DataFrame:
    """Load the training feature matrix."""
    feature_version = CFG["training"]["feature_version"]
    path = resolve_path(CFG["paths"]["feature_store"]) / f"train_detrend_{feature_version}.csv"
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_data
def load_test_features() -> pd.DataFrame:
    """Load the test feature matrix."""
    feature_version = CFG["training"]["feature_version"]
    path = resolve_path(CFG["paths"]["feature_store"]) / f"test_detrend_{feature_version}.csv"
    return pd.read_csv(path, parse_dates=["timestamp"])


@st.cache_data
def load_predictions(run_dir: str) -> pd.DataFrame:
    """Load predictions.csv for a specific inference run directory name."""
    path = _predictions_dir() / run_dir / "predictions.csv"
    return pd.read_csv(path, parse_dates=["timestamp"])


def list_inference_runs() -> list[dict]:
    """Enumerate datasets/05_predictions/ and return metadata for each run, newest first.

    Returns:
        List of dicts with keys from metadata.json plus 'dir_name'.
        Returns empty list if no runs exist or the directory is missing.
    """
    pred_dir = _predictions_dir()
    if not pred_dir.exists():
        return []

    runs = []
    for subdir in pred_dir.iterdir():
        if not subdir.is_dir():
            continue
        meta_path = subdir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            meta["dir_name"] = subdir.name
            runs.append(meta)
        except (json.JSONDecodeError, OSError):
            continue

    # Sort newest first (dir names start with YYYYMMDD_HHMMSS)
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs

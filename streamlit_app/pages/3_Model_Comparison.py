"""Page 3 — Model Training & Comparison."""
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components

from utils.mlflow_client import (
    is_mlflow_available,
    get_experiment_names,
    get_runs_summary,
    get_best_run_per_family,
    plot_image_path,
    optuna_html_path,
)
from utils.charts import model_comparison_bar

st.set_page_config(page_title="Model Comparison", page_icon=None, layout="wide")
st.title("Model Training & Comparison")
st.markdown("MLflow experiment results across all trained model families.")

# ---------------------------------------------------------------------------
# MLflow availability check
# ---------------------------------------------------------------------------
mlflow_ok = is_mlflow_available()
if not mlflow_ok:
    st.warning(
        "MLflow database not found. Showing cached summary if available. "
        "Run the training pipeline locally to populate experiment data."
    )

# ---------------------------------------------------------------------------
# Experiment selector
# ---------------------------------------------------------------------------
experiment_names = get_experiment_names() if mlflow_ok else []

if not experiment_names:
    st.info("No MLflow experiments found. Train a model to get started.")
    st.stop()

experiment = st.selectbox("Select MLflow experiment", experiment_names)

# ---------------------------------------------------------------------------
# Run summary table
# ---------------------------------------------------------------------------
st.header("All Runs")
summary = get_runs_summary(experiment)

if summary.empty or not all(c in summary.columns for c in ["model_family", "rmse", "mae"]):
    st.info("No finished runs found in this experiment.")
    st.stop()
    summary = summary  # unreachable in real Streamlit; guard for testing

display_cols = [c for c in ["model_family", "rmse", "mae", "feature_version", "run_id"] if c in summary.columns]
st.dataframe(
    summary[display_cols].rename(columns={
        "model_family": "Model Family",
        "rmse": "RMSE",
        "mae": "MAE",
        "feature_version": "Feature Version",
        "run_id": "Run ID (short)",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Best-per-family bar chart
# ---------------------------------------------------------------------------
st.header("Best Run Per Family")
best = get_best_run_per_family(experiment)

if not best.empty:
    fig = model_comparison_bar(best)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No per-family summary available.")

st.divider()

# ---------------------------------------------------------------------------
# Detailed run viewer
# ---------------------------------------------------------------------------
st.header("Run Detail")
st.markdown("Select a run to view evaluation plots.")

if not all(c in summary.columns for c in ["model_family", "run_id", "run_id_full"]):
    st.info("Run detail not available.")
    st.stop()

run_options = summary[["model_family", "run_id", "run_id_full"]].copy()
run_options["label"] = run_options["model_family"] + " — " + run_options["run_id"]
selected_label = st.selectbox("Select run", run_options["label"].tolist())

selected_row = run_options[run_options["label"] == selected_label].iloc[0]
model_family = selected_row["model_family"]
run_id_full = selected_row["run_id_full"]

# Determine run number from summary index (best-effort: try 1 then scan)
run_idx = run_options[run_options["label"] == selected_label].index[0]
run_num = int(run_idx) + 1

col_plots, col_optuna = st.columns([1, 1])

with col_plots:
    # Feature importance
    imp_path = plot_image_path(model_family, run_num, "feature_importances")
    if imp_path:
        st.subheader("Feature Importances")
        st.image(str(imp_path), use_container_width=True)
    else:
        st.info(f"No feature importance plot found for {model_family} run {run_num}.")

    # Residuals
    res_path = plot_image_path(model_family, run_num, "residuals")
    if res_path:
        st.subheader("Residuals")
        st.image(str(res_path), use_container_width=True)
    else:
        st.info(f"No residuals plot found for {model_family} run {run_num}.")

with col_optuna:
    # Correlation plot
    corr_path = plot_image_path(model_family, run_num, "correlations")
    if corr_path:
        st.subheader("Feature Correlations")
        st.image(str(corr_path), use_container_width=True)

    # Optuna optimization history
    opt_path = optuna_html_path(model_family, run_num, "optimization_history")
    if opt_path:
        st.subheader("Optuna Optimization History")
        html_content = opt_path.read_text(encoding="utf-8")
        components.html(html_content, height=450, scrolling=True)
    else:
        st.info(f"No Optuna history plot found for {model_family} run {run_num}.")

    # Param importances
    param_path = optuna_html_path(model_family, run_num, "param_importances")
    if param_path:
        st.subheader("Optuna Parameter Importances")
        html_content = param_path.read_text(encoding="utf-8")
        components.html(html_content, height=450, scrolling=True)

"""Page 4 — Forecast Results: view saved runs or trigger new inference."""
import sys
import pathlib
import datetime

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from ev_load_fc.config import CFG, resolve_path
from utils.data_loader import list_inference_runs, load_predictions
from utils.mlflow_client import (
    ALL_MODEL_FAMILIES,
    get_available_families,
    get_experiment_names,
    is_mlflow_available,
)
from utils.charts import forecast_chart

st.set_page_config(page_title="Forecast Results", page_icon=None, layout="wide")
st.title("Forecast Results")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_coverage(fc_df: pd.DataFrame, ci_level: int) -> float | None:
    lower_col = f"yhat_lower_{ci_level}"
    upper_col = f"yhat_upper_{ci_level}"
    if lower_col not in fc_df.columns or "y" not in fc_df.columns:
        return None
    mask = fc_df["y"].notna()
    if mask.sum() == 0:
        return None
    covered = (
        (fc_df.loc[mask, "y"] >= fc_df.loc[mask, lower_col]) &
        (fc_df.loc[mask, "y"] <= fc_df.loc[mask, upper_col])
    )
    return float(covered.mean())


def _run_inference(model_family: str, horizon: int, start_date: datetime.date, ci_levels: list[float]):
    """Trigger the InferencePipeline and return the forecast DataFrame."""
    from ev_load_fc.pipelines.inference_pipeline import InferencePipeline, InferencePipelineConfig
    from ev_load_fc.training.mlflow_api import init_mlflow

    # Map display name to MLflow tag value
    _FAMILY_TAG = {
        "Random Forest": "RandomForest",
        "AdaBoost": "AdaBoost",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
        "CatBoost": "CatBoost",
        "Prophet": "Prophet",
    }

    experiment_names = get_experiment_names()
    if not experiment_names:
        raise ValueError("No MLflow experiments found. Train a model first.")

    inf_cfg = CFG["inference"]
    feature_version = CFG["training"]["feature_version"]
    X_set = f"test_detrend_{feature_version}"

    config = InferencePipelineConfig(
        raw_hourly_path=resolve_path(CFG["paths"]["processed_data"]) / CFG["files"]["combined_filename"],
        feature_store=resolve_path(CFG["paths"]["feature_store"]),
        predictions_dir=resolve_path(CFG["paths"]["predictions"]),
        X_set=X_set,
        experiment_name=experiment_names[0],
        model_family=_FAMILY_TAG.get(model_family, model_family),
        feature_version=feature_version,
        metric=inf_cfg.get("metric", "rmse"),
        horizon=horizon,
        inference_start=pd.Timestamp(start_date),
        confidence_intervals=inf_cfg.get("confidence_intervals", ci_levels),
        n_bootstrap=inf_cfg.get("n_bootstrap", 50),
    )
    pipeline = InferencePipeline(config)
    return pipeline.run()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_view, tab_run = st.tabs(["View Saved Forecasts", "Run New Forecast"])

# ===========================================================================
# Tab 1 — View saved forecasts
# ===========================================================================
with tab_view:
    runs = list_inference_runs()

    if not runs:
        st.info(
            "No saved forecasts yet. Use the **Run New Forecast** tab to generate one."
        )
    else:
        # Build selectbox labels
        def _label(r: dict) -> str:
            created = r.get("created_at", "")[:16].replace("T", " ")
            family = r.get("model_family", "Unknown")
            horizon = r.get("horizon", "?")
            start = r.get("inference_start", "")[:10]
            return f"{created}  |  {family}  |  {horizon}h  |  from {start}"

        labels = [_label(r) for r in runs]
        selected_label = st.selectbox("Select forecast run", labels)
        selected_run = runs[labels.index(selected_label)]

        # CI level selector
        ci_options = {
            "80%": 80,
            "95%": 95,
            "Both (80% shown)": 80,
        }
        ci_choice = st.radio("Confidence interval", list(ci_options.keys()), horizontal=True)
        ci_level = ci_options[ci_choice]

        # Load and display
        fc_df = load_predictions(selected_run["dir_name"])

        # Metrics row
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        metrics = selected_run.get("metrics", {})
        m_col1.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.3f}" if metrics.get("rmse") else "N/A")
        m_col2.metric("MAE", f"{metrics.get('mae', 'N/A'):.3f}" if metrics.get("mae") else "N/A")
        coverage = _compute_coverage(fc_df, ci_level)
        m_col3.metric(f"{ci_level}% Coverage", f"{coverage:.1%}" if coverage is not None else "N/A")
        m_col4.metric("Horizon", f"{selected_run.get('horizon', '?')}h")

        # Chart
        fig = forecast_chart(fc_df, ci_level=ci_level)
        st.plotly_chart(fig, use_container_width=True)

        # Also show 95% ribbon if "Both" selected
        if ci_choice == "Both (80% shown)" and "yhat_lower_95" in fc_df.columns:
            st.caption("95% CI band:")
            fig_95 = forecast_chart(fc_df, ci_level=95)
            st.plotly_chart(fig_95, use_container_width=True)

        # Download
        csv_bytes = fc_df.to_csv(index=False).encode()
        st.download_button(
            label="Download forecast CSV",
            data=csv_bytes,
            file_name=f"forecast_{selected_run['dir_name']}.csv",
            mime="text/csv",
        )

# ===========================================================================
# Tab 2 — Run new forecast
# ===========================================================================
with tab_run:
    mlflow_ok = is_mlflow_available()

    if not mlflow_ok:
        st.error(
            "MLflow is not available. Run the training pipeline locally "
            "before triggering inference."
        )
        st.stop()

    # Model family selector
    experiment_names = get_experiment_names()
    available_families = get_available_families(experiment_names[0]) if experiment_names else []

    st.subheader("Inference Parameters")

    col_left, col_right = st.columns(2)

    with col_left:
        selected_family = st.selectbox(
            "Model family",
            ALL_MODEL_FAMILIES,
            help="Only families with a trained MLflow run can be used.",
        )

        if selected_family not in available_families:
            st.warning(
                f"No trained runs found for **{selected_family}**. "
                f"Available: {', '.join(available_families) or 'none'}."
            )

        horizon = st.select_slider(
            "Forecast horizon (hours)",
            options=[24, 48, 72, 168, 720, 2160],
            value=168,
        )

    with col_right:
        # Constrain to test set range from config
        test_start = pd.Timestamp(CFG["data"]["preprocessing"]["split_date"]).date()
        data_end = pd.Timestamp(CFG["data"]["raw_filters"]["max_timestamp"]).date()

        start_date = st.date_input(
            "Forecast start date",
            value=test_start,
            min_value=test_start,
            max_value=data_end,
            help="Must be within the test period (after the train/test split).",
        )

        ci_choice_run = st.radio(
            "Confidence interval level",
            ["80%", "95%", "Both"],
            horizontal=True,
        )
        ci_map = {"80%": [0.80], "95%": [0.95], "Both": [0.80, 0.95]}
        ci_levels = ci_map.get(str(ci_choice_run), [0.80, 0.95])

    st.divider()

    run_disabled = selected_family not in available_families
    if st.button(
        "Run Forecast",
        type="primary",
        disabled=run_disabled,
        help="Disabled if the selected model family has no trained runs." if run_disabled else "",
    ):
        with st.spinner(f"Running {horizon}h recursive forecast with {selected_family}..."):
            try:
                fc_df = _run_inference(selected_family, horizon, start_date, ci_levels)
                st.success("Forecast complete! Switch to **View Saved Forecasts** to see the result.")

                # Preview the result immediately
                ci_display = int(ci_levels[0] * 100)
                fig = forecast_chart(fc_df, ci_level=ci_display)
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                if "y" in fc_df.columns and fc_df["y"].notna().any():
                    mask = fc_df["y"].notna()
                    residuals = fc_df.loc[mask, "y"] - fc_df.loc[mask, "yhat"]
                    rmse = float(np.sqrt((residuals ** 2).mean()))
                    mae = float(residuals.abs().mean())
                    mc1, mc2 = st.columns(2)
                    mc1.metric("RMSE", f"{rmse:.3f} kWh")
                    mc2.metric("MAE", f"{mae:.3f} kWh")

            except Exception as e:
                st.error(f"Forecast failed: {e}")

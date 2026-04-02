"""Page 2 — Data & Exploratory Data Analysis."""
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from ev_load_fc.config import CFG
from utils.data_loader import load_processed
from utils.charts import (
    energy_timeseries,
    hourly_seasonality,
    weekly_seasonality,
    correlation_bar,
)

st.set_page_config(page_title="Data & EDA", page_icon=None, layout="wide")
st.title("Data & Exploratory Analysis")
st.markdown("Interactive exploration of the hourly EV charging load dataset.")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Loading processed data..."):
    df = load_processed()

# Parse train/test split from config
train_cutoff = pd.Timestamp(CFG["data"]["preprocessing"]["split_date"])

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

date_min = df["timestamp"].min().date()
date_max = df["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max,
)

show_outliers = st.sidebar.toggle("Highlight outliers", value=False)

# Apply date filter
if len(date_range) == 2:
    try:
        start, end = date_range
        start = pd.Timestamp(start).date()
        end = pd.Timestamp(end).date()
        mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
        df_filtered = df[mask].copy()
    except Exception:
        df_filtered = df.copy()
else:
    df_filtered = df.copy()

# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------
st.header("Dataset Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows (filtered)", f"{len(df_filtered):,}")
col2.metric("Avg Energy (kWh)", f"{df_filtered['energy'].mean():.2f}")
col3.metric("Max Energy (kWh)", f"{df_filtered['energy'].max():.2f}")
col4.metric("Outlier %", f"{(df_filtered['energy_outlier'].mean() * 100):.1f}%" if "energy_outlier" in df_filtered.columns else "N/A")

st.divider()

# ---------------------------------------------------------------------------
# Energy time series
# ---------------------------------------------------------------------------
st.header("Energy Time Series")
fig_ts = energy_timeseries(df_filtered, train_cutoff, show_outliers=show_outliers)
st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------
st.header("Seasonality Patterns")
col_hour, col_week = st.columns(2)

with col_hour:
    st.plotly_chart(hourly_seasonality(df_filtered), use_container_width=True)

with col_week:
    st.plotly_chart(weekly_seasonality(df_filtered), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Feature correlations
# ---------------------------------------------------------------------------
st.header("Feature Correlations with Energy")
st.markdown("Pearson correlation of all numeric features with the energy target.")

numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()
if "energy" in numeric_cols and len(numeric_cols) > 1:
    fig_corr = correlation_bar(df_filtered, target_col="energy")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Not enough numeric features to compute correlations.")

st.divider()

# ---------------------------------------------------------------------------
# Weather & traffic features (expandable)
# ---------------------------------------------------------------------------
weather_cols = [c for c in df_filtered.columns if c.startswith("w_")]
traffic_cols = [c for c in df_filtered.columns if c.startswith("t_")]

if weather_cols:
    with st.expander("Weather Event Features", expanded=False):
        selected_weather = st.multiselect(
            "Select weather features to plot",
            weather_cols,
            default=weather_cols[:3],
        )
        if selected_weather:
            import plotly.graph_objects as go
            fig_w = go.Figure()
            for col in selected_weather:
                fig_w.add_trace(go.Scatter(
                    x=df_filtered["timestamp"],
                    y=df_filtered[col],
                    mode="lines",
                    name=col,
                ))
            fig_w.update_layout(
                title="Weather Features Over Time",
                xaxis_title="Date",
                yaxis_title="Duration (hours)",
                template="plotly_white",
                height=380,
                hovermode="x unified",
            )
            st.plotly_chart(fig_w, use_container_width=True)

if traffic_cols:
    with st.expander("Traffic Event Features", expanded=False):
        selected_traffic = st.multiselect(
            "Select traffic features to plot",
            traffic_cols,
            default=traffic_cols[:3],
        )
        if selected_traffic:
            import plotly.graph_objects as go
            fig_t = go.Figure()
            for col in selected_traffic:
                fig_t.add_trace(go.Scatter(
                    x=df_filtered["timestamp"],
                    y=df_filtered[col],
                    mode="lines",
                    name=col,
                ))
            fig_t.update_layout(
                title="Traffic Features Over Time",
                xaxis_title="Date",
                template="plotly_white",
                height=380,
                hovermode="x unified",
            )
            st.plotly_chart(fig_t, use_container_width=True)

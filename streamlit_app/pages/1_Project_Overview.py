"""Page 1 — Project Overview."""
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(page_title="Project Overview", page_icon=None, layout="wide")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Project Overview")
st.markdown(
    """
    *A machine learning pipeline for forecasting electric vehicle (EV) charging load
    in Palo Alto, California — built as a personal portfolio project.*
    """
)

# Tech stack badges (using markdown-rendered pill style)
st.markdown(
    """
    `Python 3.11` &nbsp; `LightGBM` &nbsp; `CatBoost` &nbsp; `XGBoost` &nbsp;
    `scikit-learn` &nbsp; `Prophet` &nbsp; `MLflow` &nbsp; `Optuna` &nbsp;
    `Pandas` &nbsp; `Streamlit`
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Problem statement
# ---------------------------------------------------------------------------
st.header("Problem Statement")
st.markdown(
    """
    As EV adoption accelerates, accurate short-term load forecasting is critical for grid
    operators and charging network managers. This project builds a **recursive multi-step
    forecasting pipeline** that predicts hourly EV charging energy (kWh) up to one week
    ahead using historical charging data, weather events, traffic conditions, and
    engineered temporal features.

    The forecast horizon is configurable (24h to 168h) and confidence intervals are
    estimated from ensemble variance — enabling uncertainty-aware planning.
    """
)

st.divider()

# ---------------------------------------------------------------------------
# Pipeline diagram
# ---------------------------------------------------------------------------
st.header("Pipeline Architecture")
st.code(
    """
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        EV Load Forecasting Pipeline                     │
  └─────────────────────────────────────────────────────────────────────────┘

  Raw Data (ChargePoint + Weather + Traffic)
       │
       ▼
  ┌──────────────┐    ┌─────────────────────┐    ┌───────────────────────┐
  │  Extraction  │───▶│   Preprocessing     │───▶│  Feature Engineering  │
  │  (CSV → DB)  │    │  (clean, detrend,   │    │  (lags, rolling win,  │
  └──────────────┘    │   train/test split) │    │   time, holidays)     │
                      └─────────────────────┘    └───────────────────────┘
                                                           │
                                                           ▼
                                              ┌────────────────────────┐
                                              │  Feature Selection     │
                                              │  (F-test k=30 → RFE   │
                                              │   k=20)                │
                                              └────────────────────────┘
                                                           │
                                                           ▼
                                   ┌───────────────────────────────────────┐
                                   │  Optuna Hyperparameter Optimisation   │
                                   │  Time-series CV (4 splits)            │
                                   │  Models: RF, AdaBoost, XGB, LGBM,     │
                                   │          CatBoost, Prophet            │
                                   └───────────────────────────────────────┘
                                                           │
                                                           ▼
                                              ┌────────────────────────┐
                                              │   MLflow Tracking      │
                                              │  (metrics, params,     │
                                              │   artifacts, plots)    │
                                              └────────────────────────┘
                                                           │
                                                           ▼
                                   ┌───────────────────────────────────────┐
                                   │  Recursive Inference Pipeline         │
                                   │  • Load best model from MLflow        │
                                   │  • Predict one step → update lags     │
                                   │  • Repeat for full horizon            │
                                   │  • Compute CI via ensemble variance   │
                                   └───────────────────────────────────────┘
""",
    language=None,
)

st.divider()

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------
st.header("Dataset at a Glance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Hourly Records", "21,936")
col2.metric("Date Range", "Aug 2017 – Feb 2020")
col3.metric("Train / Test Split", "Aug 2019")
col4.metric("Model Families Evaluated", "6")

st.divider()

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
st.header("Data Sources")
col_a, col_b, col_c, col_d = st.columns(4)

with col_a:
    st.markdown("#### EV Charging")
    st.markdown(
        """
        **ChargePoint CY20Q4**
        Hourly-aggregated EV charging sessions from Palo Alto public
        charge points. Target: total energy (kWh) per hour.
        """
    )

with col_b:
    st.markdown("#### Weather Events")
    st.markdown(
        """
        **NOAA Weather Events**
        Rain, fog, and storm events with severity ratings for the
        Mountain View / Palo Alto area.
        """
    )

with col_c:
    st.markdown("#### Traffic Events")
    st.markdown(
        """
        **US Traffic Events**
        Congestion and flow-incident events with duration and
        distance impact, matched to the local road network.
        """
    )

with col_d:
    st.markdown("#### Temperature")
    st.markdown(
        """
        **Meteostat API**
        Hourly temperature data for Palo Alto, with imputed values
        for missing observations.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Feature engineering summary
# ---------------------------------------------------------------------------
st.header("Feature Engineering")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(
        """
        **Temporal features**
        - Sinusoidal encodings: hour, weekday, month
        - US Federal Holiday one-hot encodings

        **Lag features (energy)**
        - Lags: 1h, 3h, 6h, 24h, 168h (1 week)

        **Rolling windows (energy)**
        - Sums and means over 3h, 6h, 12h, 24h, 168h
        """
    )

with col_right:
    st.markdown(
        """
        **Weather features**
        - Rolling sums of fog/rain/storm duration (1h, 3h, 6h)
        - Severity-weighted aggregations

        **Traffic features**
        - Congestion and flow-incident duration/distance

        **Feature selection**
        - Stage 1: F-test → top 30 features
        - Stage 2: RFE → top 20 features
        """
    )

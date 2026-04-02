"""EV Load Forecasting — Streamlit portfolio app entry point."""
import sys
import pathlib

# Make the project package importable when running from streamlit_app/
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="EV Load Forecasting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("EV Load Forecasting")
st.sidebar.markdown(
    """
    A machine learning portfolio project demonstrating end-to-end time series forecasting
    for EV charging load in Palo Alto, CA.

    ---
    **Navigate using the pages above.**
    """
)

st.title("EV Load Forecasting")
st.markdown(
    """
    Welcome! Use the sidebar to navigate between sections of this portfolio project.

    | Section | Description |
    |---------|-------------|
    | **Project Overview** | Background, pipeline architecture, and dataset summary |
    | **Data & EDA** | Interactive exploration of the hourly energy time series |
    | **Model Comparison** | MLflow experiment results across 6 model families |
    | **Forecast Results** | View saved forecasts or run new recursive predictions |
    """
)

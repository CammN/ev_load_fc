"""EV Load Forecasting — Streamlit portfolio app entry point."""
import sys
import pathlib

# Make the project package importable when running from streamlit_app/
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="Electric Vehicle Supply Equipment (EVSE) Load Forecasting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.sidebar.title("EVSE Load Forecasting")
# st.sidebar.markdown(
#     """
#     A machine learning portfolio project demonstrating end-to-end time series forecasting
#     for EV charging load in Palo Alto, CA.

#     ---
#     **Navigate using the pages above.**
#     """
# )

st.title("Electric Vehicle Supply Equipment (EVSE) Load Forecasting")
st.markdown(
    """

    This project aims to use machine learning to model hourly demand for EVSE's in the Palo Alto area using historical data. \\
    The data covers the period August 2016 - December 2020 and comes from 4 sources:  

    - EVSE charging events data - City of Palo Alto Open Data programme
    - Traffic events data - MapQuest public API, curated into the LSTW: Large-Scale Traffic and Weather Events Dataset
    - Weather events data - Meteostat public API, curated into the LSTW: Large-Scale Traffic and Weather Events Dataset
    - Temperature data - Meteostat public API

    .......

    ## Navigation

    | Section | Description |
    |---------|-------------|
    | **Project Overview** | Background, pipeline architecture, and dataset summary |
    | **Data & EDA** | Interactive exploration of the hourly energy time series |
    | **Model Comparison** | MLflow experiment results across 6 model families |
    | **Forecast Results** | View saved forecasts or run new recursive predictions |
    """
)

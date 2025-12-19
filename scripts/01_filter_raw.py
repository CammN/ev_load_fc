# This script allows a user to read in and filter 1-3 of the raw datasets based on date ranges and state/county/city
# Additional temperature data is brought in via the meteostat public API and merged onto the weather dataset
# It then standardises all column names and saves as CSV files

import argparse
import logging
import pandas as pd
from ev_load_fc.pipelines.loading_pipeline import LoadingPipeline, LoadingPipelineConfig
from ev_load_fc.config import CFG, resolve_path
from ev_load_fc.utils.logging import setup_logging
logging_level = CFG["project"]["logging_level"]


def build_pipeline_params(ev, weather, traffic):

    raw = resolve_path(CFG["paths"]["raw_data"])
    interim = resolve_path(CFG["paths"]["interim_data"])

    return {
        # Paths
        "ev_raw_path": raw / CFG["files"]["ev_filename"],
        "weather_raw_path": raw / CFG["files"]["weather_filename"],
        "traffic_raw_path": raw / CFG["files"]["traffic_filename"],
        "ev_int_path": interim / CFG["files"]["ev_filt_filename"],
        "weather_int_path": interim / CFG["files"]["weather_filt_filename"],
        "traffic_int_path": interim / CFG["files"]["traffic_filt_filename"],
        "temp_path": interim / CFG["files"]["temperature_filename"],
        # Filters
        "min_timestamp": pd.to_datetime(CFG["data"]["raw_filters"]["min_timestamp"]),
        "max_timestamp": pd.to_datetime(CFG["data"]["raw_filters"]["max_timestamp"]),
        "weather_cities": CFG["data"]["raw_filters"]["weather_cities"],
        "traffic_cities": CFG["data"]["raw_filters"]["traffic_cities"],
        "mts_stations": CFG["data"]["raw_filters"]["meteostat_staions"],
        # Pipeline parameters
        "run_ev": ev,
        "run_weather": weather,
        "run_traffic": traffic,
    }


def parse_args():
    """Parse command line arguments for processing raw datasets."""
    parser = argparse.ArgumentParser(description="Trim raw EV, weather, and traffic data into interim datasets.")
    parser.add_argument( "--ev", action="store_true", help="Process EV data.")
    parser.add_argument("--weather", action="store_true", help="Process weather data.")
    parser.add_argument( "--traffic", action="store_true", help="Process traffic data.")

    return parser.parse_args()


def main():
    logger = setup_logging("preprocessing_pipeline.log", level=logging_level)
    args = parse_args()

    # If no flags are provided in CLI, run everything
    if not (args.ev or args.weather or args.traffic):
        args.ev = args.weather = args.traffic = True

    # Build pipeline parameters (including run parameters) and convert to dataclass
    pipeline_params = build_pipeline_params(
        ev=args.ev,
        weather=args.weather,
        traffic=args.traffic,
    )
    cfg = LoadingPipelineConfig(**pipeline_params)

    pipeline = LoadingPipeline(config=cfg)
    logger.info("Starting LoadingPipeline...")
    pipeline.run()
    logger.info("Finished LoadingPipeline.")
    logger.info("...")


if __name__ == "__main__":
    main()

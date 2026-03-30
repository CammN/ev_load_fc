import argparse
import logging
import pandas as pd
from ev_load_fc.pipelines.preprocessing_pipeline import PreprocessingPipeline, PreprocessingPipelineConfig
from ev_load_fc.config import CFG, resolve_path
from ev_load_fc.utils.logging import setup_logging
logging_level = CFG["project"]["logging_level"]

def build_pipeline_params(ev, weather, temperature, traffic, combine):

    interim = resolve_path(CFG["paths"]["interim_data"])
    processed = resolve_path(CFG["paths"]["processed_data"])

    return {

        # Paths
        "processed_data_path": processed,
        "ev_int_path": interim / CFG["files"]["ev_filt_filename"],
        "weather_int_path": interim / CFG["files"]["weather_filt_filename"],
        "temp_path": interim / CFG["files"]["temperature_filename"],
        "traffic_int_path": interim / CFG["files"]["traffic_filt_filename"],
        "ev_proc_path": processed / CFG["files"]["ev_proc_filename"],
        "weather_proc_path": processed / CFG["files"]["weather_proc_filename"],
        "temp_proc_path": processed / CFG["files"]["temperature_proc_filename"],
        "traffic_proc_path": processed / CFG["files"]["traffic_proc_filename"],
        "combined_path": processed / CFG["files"]["combined_filename"],

        # Filters
        "min_timestamp": pd.to_datetime(CFG["data"]["raw_filters"]["min_timestamp"]),
        "max_timestamp": pd.to_datetime(CFG["data"]["raw_filters"]["max_timestamp"]),
        "ev_cols": CFG["data"]["preprocessing"]["ev_keep_cols"],
        "weather_cols": CFG["data"]["preprocessing"]["weather_keep_cols"],
        "temp_cols": [],
        "traffic_cols": CFG["data"]["preprocessing"]["traffic_keep_cols"],
        "weather_type_filt": CFG["data"]["preprocessing"]["weather_type_filt"],
        "traffic_type_filt": CFG["data"]["preprocessing"]["traffic_type_filt"],

        # Preprocessing parameters
        "kw_quant": CFG["data"]["preprocessing"]["plug_power_quantile_bound"],
        "mad_thresh": CFG["data"]["preprocessing"]["mad_thresholds"],
        "split_date": pd.to_datetime(CFG["data"]["preprocessing"]["split_date"]),
        "sampling_interval": CFG["data"]["preprocessing"]["sampling_interval"],
        "min_outlier_samples": CFG["data"]["preprocessing"]["min_outlier_samples"],
        
        # Optional runtime parameters
        "run_ev": ev,
        "run_weather": weather,
        "run_temperature": temperature,
        "run_traffic": traffic,
        "run_combine": combine,
    }

def parse_args():
    """Parse command line arguments for processing interim datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument( "--ev", action="store_true", help="Process EV data.")
    parser.add_argument("--weather", action="store_true", help="Process weather data.")
    parser.add_argument("--temperature", action="store_true", help="Process temperature data.")
    parser.add_argument( "--traffic", action="store_true", help="Process traffic data.")
    parser.add_argument( "--combine", action="store_true", help="Combine all data.")

    return parser.parse_args()

def main():
    # Initialise logging and CLI args
    logger = setup_logging("preprocessing_pipeline.log", level=logging_level)
    args = parse_args()

    # If no flags are provided in CLI, run everything
    if not (args.ev or args.weather or args.temperature or args.traffic or args.combine):
        args.ev = args.weather = args.temperature = args.traffic = args.combine = True

    # Build pipeline parameters (including run parameters) and convert to dataclass
    pipeline_params = build_pipeline_params(
        ev=args.ev,
        weather=args.weather,
        temperature=args.temperature,
        traffic=args.traffic,
        combine=args.combine,
    )
    cfg = PreprocessingPipelineConfig(**pipeline_params)

    pipeline = PreprocessingPipeline(config=cfg)
    logger.info("Starting PreprocessingPipeline...")
    pipeline.run()
    logger.info("Finished PreprocessingPipeline.")
    logger.info("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()

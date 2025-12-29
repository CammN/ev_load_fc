import argparse
import logging
import pandas as pd
from ev_load_fc.config import CFG, resolve_path
from ev_load_fc.utils.logging import setup_logging
from ev_load_fc.pipelines.feature_pipeline import FeaturePipeline, FeaturePipelineConfig
logging_level = CFG["project"]["logging_level"]


def build_pipeline_params(fe, fs):

    processed = resolve_path(CFG["paths"]["processed_data"])

    return {
        # Paths
        "combined_path": processed / CFG["files"]["combined_filename"],
        "feature_store": resolve_path(CFG["paths"]["feature_store"]),

        # Miscellaneous
        "min_timestamp": pd.to_datetime(CFG["data"]["raw_filters"]["min_timestamp"]),
        "max_timestamp": pd.to_datetime(CFG["data"]["raw_filters"]["max_timestamp"]),
        "split_date": pd.to_datetime(CFG["data"]["preprocessing"]["split_date"]),
        "seed": CFG["project"]["seed"],

        # Feature creation parameters
        "target": CFG["features"]["feature_engineering"]["target"],
        "holiday_list": list(CFG["features"]["feature_engineering"]["holidays"]),
        "time_feature_dict": CFG["features"]["feature_engineering"]["time_feature_dict"],
        "energy_col_substrs": CFG["features"]["feature_engineering"]["energy_col_substrs"],
        "weather_col_substrs": CFG["features"]["feature_engineering"]["weather_col_substrs"],
        "temperature_col_substrs": CFG["features"]["feature_engineering"]["temperature_col_substrs"],
        "traffic_col_substrs": CFG["features"]["feature_engineering"]["traffic_col_substrs"],
        
        # Feature selection parameters
        "scale_method": CFG["features"]["feature_selection"]["scale_method"],
        "k_1": CFG["features"]["feature_selection"]["k_1"],
        "fe_method_1": CFG["features"]["feature_selection"]["fe_method_1"],
        "k_2": CFG["features"]["feature_selection"]["k_2"],
        "fe_method_2": CFG["features"]["feature_selection"]["fe_method_2"],
        "corr_thresh": CFG["features"]["feature_selection"]["correlation_threshold"],
        "holidays": set(CFG["features"]["feature_engineering"]["holidays"]),

        # Optional runtime parameters
        "run_feat_eng": fe,
        "run_feat_select": fs,
    }

def parse_args():
    """Parse command line arguments for running the feature pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument( "--fe", action="store_true", help="Perform feature engineering.")
    parser.add_argument( "--fs", action="store_true", help="Perform feature selection.")

    return parser.parse_args()


def main():
    # Initialise logging and CLI args
    logger = setup_logging("feature_pipeline.log", level=logging_level)
    args = parse_args()

    # If no flags are provided in CLI, run everything
    if not (args.fe) and not (args.fs):
        args.fe = args.fs = True

    # Build pipeline parameters (including run parameters) and convert to dataclass
    pipeline_params = build_pipeline_params(
        args.fe,
        args.fs,
    )
    cfg = FeaturePipelineConfig(**pipeline_params)

    pipeline = FeaturePipeline(config=cfg)
    logger.info("Starting FeaturePipeline...")
    pipeline.run()
    logger.info("Finished FeaturePipeline.")
    logger.info("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()

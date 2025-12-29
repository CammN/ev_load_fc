import argparse
import logging
import pandas as pd
from ev_load_fc.config import CFG, resolve_path
from ev_load_fc.utils.logging import setup_logging
from ev_load_fc.pipelines.training_pipeline import TrainingPipeline, TrainingPipelineConfig
logging_level = CFG["project"]["logging_level"]


def build_pipeline_params(fs):

    processed = resolve_path(CFG["paths"]["processed_data"])

    return {
        # Paths
        "feature_store":  resolve_path(CFG["paths"]["feature_store"]),
        # Pre training parameters
        "scale_method": CFG["features"]["feature_selection"]["scale_method"],
        "k_1": CFG["features"]["feature_selection"]["k_1"],
        "fe_method_1": CFG["features"]["feature_selection"]["fe_method_1"],
        "k_2": CFG["features"]["feature_selection"]["k_2"],
        "fe_method_2": CFG["features"]["feature_selection"]["fe_method_2"],
        "corr_thresh": CFG["features"]["feature_selection"]["correlation_threshold"],
        "holidays": set(CFG["features"]["feature_engineering"]["holidays"]),
        # Other
        "split_date": pd.to_datetime(CFG["data"]["preprocessing"]["split_date"]),
        "seed": CFG["project"]["seed"],
        # Optional runtime parameters
        "run_feat_select": fs,
    }


def parse_args():
    """Parse command line arguments for running the feature selection and model training pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument( "--fs", action="store_true", help="Perform feature selection.")

    return parser.parse_args()


def main():
    # Initialise logging and CLI args
    logger = setup_logging("training_pipeline.log", level=logging_level)
    args = parse_args()

    # If no flags are provided in CLI, run everything
    if not (args.fs):
        args.fs = True

    # Build pipeline parameters (including run parameters) and convert to dataclass
    pipeline_params = build_pipeline_params(
        args.fs
    )
    cfg = TrainingPipelineConfig(**pipeline_params)

    pipeline = TrainingPipeline(config=cfg)
    logger.info("Starting TrainingPipeline...")
    pipeline.run()
    logger.info("Finished TrainingPipeline.")
    logger.info("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()

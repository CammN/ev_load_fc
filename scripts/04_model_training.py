import argparse
import logging
from ev_load_fc.config import CFG, resolve_path
from ev_load_fc.utils.logging import setup_logging
from ev_load_fc.pipelines.training_pipeline import TrainingPipeline, TrainingPipelineConfig
from ev_load_fc.training.registry import MODEL_REGISTRY
logger = logging.getLogger(__name__)
logging_level = CFG["project"]["logging_level"]


def build_pipeline_params():

    # processed = resolve_path(CFG["paths"]["processed_data"])

    return {
        # Paths
        "feature_store": resolve_path(CFG["paths"]["feature_store"]),
        "configs": resolve_path(CFG["paths"]["configs"]),
        # MLFlow parameters
        "tracking_uri": CFG["training"]["mlflow"]["tracking_uri"],
        "experiment_name": CFG["training"]["mlflow"]["experiment_name"],
        # Optuna parameters
        "verbosity": CFG["training"]["optuna"]["verbosity"],
        "models_to_run": CFG["training"]["optuna"]["models_to_run"],
        "trials": CFG["training"]["optuna"]["trials"],
        "metric": CFG["training"]["optuna"]["metric"],
        "splits": CFG["training"]["optuna"]["splits"],
        "search_spaces": CFG["training"]["optuna"]["search_spaces"],
        # Miscellaneous
        "seed": CFG["project"]["seed"],
        "target": CFG["features"]["target"],
        "feature_version": CFG["training"]["feature_version"]
    }


# def parse_args():
#     """Parse command line arguments for running the model training pipeline."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument( "--fs", action="store_true", help="Perform feature selection.")

#     return parser.parse_args()


def main():
    # Initialise logging and CLI args
    logger = setup_logging("training_pipeline.log", level=logging_level)
    # args = parse_args()

    # If no flags are provided in CLI, run everything
    # if not (args.fs):
    #     args.fs = True

    # Build pipeline parameters (including run parameters) and convert to dataclass
    pipeline_params = build_pipeline_params()
    cfg = TrainingPipelineConfig(**pipeline_params)

    pipeline = TrainingPipeline(config=cfg)
    logger.info("Starting TrainingPipeline...")
    pipeline.run()
    logger.info("Finished TrainingPipeline.")
    logger.info("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()

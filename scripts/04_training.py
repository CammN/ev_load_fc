import argparse
import logging
from typing import List
from ev_load_fc.config import CFG, resolve_path
from ev_load_fc.utils.logging import setup_logging
from ev_load_fc.pipelines.training_pipeline import TrainingPipeline, TrainingPipelineConfig
logger = logging.getLogger(__name__)
logging_level = CFG["project"]["logging_level"]


def build_pipeline_params(exp_name:str|None=None, model_names:List[str]|None=None):

    # If we pass a different experiment name or a set of models to train then override what we have in the config file
    experiment_name = exp_name or CFG["training"]["mlflow"]["experiment_name"]
    models_to_run = model_names or CFG["training"]["optuna"]["models_to_run"]

    return {
        # Paths and files
        "feature_store": resolve_path(CFG["paths"]["feature_store"]),
        "configs": resolve_path(CFG["paths"]["configs"]),
        "images": resolve_path(CFG["paths"]["images"]),
        "train": CFG['files']['train'],
        "test": CFG['files']['test'],
        # MLFlow parameters
        "experiment_name": experiment_name,
        # Optuna parameters
        "verbosity": CFG["training"]["optuna"]["verbosity"],
        "models_to_run": models_to_run,
        "trials": CFG["training"]["optuna"]["trials"],
        "metric": CFG["training"]["optuna"]["metric"],
        "splits": CFG["training"]["optuna"]["splits"],
        "search_spaces": CFG["training"]["optuna"]["search_spaces"],
        # Miscellaneous
        "seed": CFG["project"]["seed"],
        "target": CFG["features"]["target"],
        "feature_version": CFG["training"]["feature_version"]
    }


def parse_args():
    """Parse command line arguments for running the model training pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument( "--exp_name", nargs="+", type=str, help="Select MLflow experiment by name (overrides config).")
    parser.add_argument( "--model_names", nargs="+", type=str, help="Select models to run in MLlow experiment (overrides config).")

    return parser.parse_args()


def main():
    # Initialise logging and CLI args
    logger = setup_logging("training_pipeline.log", level=logging_level)
    args = parse_args()

    # Build pipeline parameters (including run parameters) and convert to dataclass
    pipeline_params = build_pipeline_params(
        exp_name = args.exp_name,
        model_names = args.model_names
    )
    cfg = TrainingPipelineConfig(**pipeline_params)

    pipeline = TrainingPipeline(config=cfg)
    logger.info("Starting TrainingPipeline...")
    pipeline.run()
    logger.info("Finished TrainingPipeline.")
    logger.info("-----------------------------------------------------------------")


if __name__ == "__main__":
    main()

import argparse
import logging
import pandas as pd
from ev_load_fc.config import CFG, resolve_path
from ev_load_fc.utils.logging import setup_logging
from ev_load_fc.pipelines.inference_pipeline import InferencePipeline, InferencePipelineConfig
logger = logging.getLogger(__name__)
logging_level = CFG["project"]["logging_level"]


def build_pipeline_params(
    horizon: int | None = None,
    inference_start: str | None = None,
    model_family: str | None = None,
    feature_version: str | None = None,
    metric: str | None = None,
) -> dict:
    """Build InferencePipelineConfig kwargs from config.yaml with optional CLI overrides."""

    inf = CFG["inference"]

    # CLI args take priority over config values
    resolved_horizon = horizon if horizon is not None else inf["horizon"]
    resolved_start = pd.Timestamp(inference_start or inf["inference_start"])
    resolved_model_family = model_family if model_family is not None else inf.get("model_family", "")
    resolved_feature_version = feature_version if feature_version is not None else inf.get("feature_version", "")
    resolved_metric = metric if metric is not None else inf["metric"]

    return {
        # Paths
        "raw_hourly_path": resolve_path(inf["raw_hourly"]),
        "feature_store": resolve_path(CFG["paths"]["feature_store"]),
        "predictions_dir": resolve_path(CFG["paths"]["predictions"]),
        "X_set": inf["X_set"],
        # MLflow model selection
        "experiment_name": inf["experiment_name"],
        "model_family": resolved_model_family,
        "feature_version": resolved_feature_version,
        "metric": resolved_metric,
        # Forecast settings
        "horizon": resolved_horizon,
        "inference_start": resolved_start,
        # CI settings
        "confidence_intervals": inf["confidence_intervals"],
        "n_bootstrap": inf["n_bootstrap"],
    }


def parse_args():
    """Parse command line arguments for the inference pipeline."""
    parser = argparse.ArgumentParser(description="Run EV load recursive forecast inference.")
    parser.add_argument(
        "--horizon", type=int,
        help="Number of steps to forecast (overrides config)."
    )
    parser.add_argument(
        "--inference_start", type=str,
        help="Forecast start timestamp, e.g. '2019-10-01 00:00:00' (overrides config)."
    )
    parser.add_argument(
        "--model_family", type=str,
        help="Filter MLflow runs by model family tag, e.g. 'XGBoost' (overrides config)."
    )
    parser.add_argument(
        "--feature_version", type=str,
        help="Filter MLflow runs by feature set version tag, e.g. 'E_f_30_rfe_20' (overrides config)."
    )
    parser.add_argument(
        "--metric", type=str, choices=["rmse", "mae"],
        help="Metric used to rank and select the best run: 'rmse' or 'mae' (overrides config)."
    )
    return parser.parse_args()


def main():
    logger = setup_logging("inference_pipeline.log", level=logging_level)
    args = parse_args()

    pipeline_params = build_pipeline_params(
        horizon=args.horizon,
        inference_start=args.inference_start,
        model_family=args.model_family,
        feature_version=args.feature_version,
        metric=args.metric,
    )
    cfg = InferencePipelineConfig(**pipeline_params)

    pipeline = InferencePipeline(config=cfg)
    logger.info("Starting InferencePipeline...")
    fc_df = pipeline.run()
    logger.info(f"InferencePipeline complete. Forecast shape: {fc_df.shape}")
    logger.info("-----------------------------------------------------------------")

    return fc_df


if __name__ == "__main__":
    main()
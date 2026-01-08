import pandas as pd
import pathlib
import mlflow
import optuna
from dataclasses import dataclass
from ev_load_fc.training.mlflow_api import get_or_create_experiment, parent_logging
from ev_load_fc.training.optuna_api import (
    prophet_df_format,
    cv_score_prophet_model,
    objective,
    champion_callback,

)
import logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingPipelineConfig:
    # Paths
    feature_store: pathlib.Path
    configs: pathlib.Path
    # MLFlow parameters
    tracking_uri: str
    experiment_name: str
    # Optuna parameters
    verbosity: int
    models_to_run: list
    trials: int
    metric: str
    splits: int
    search_spaces: dict
    # Miscellaneous
    seed: int
    target: str
    feature_version: str


class TrainingPipeline:


    def __init__(self, config: TrainingPipelineConfig):
        self.cfg = config


    def _load_data(self):
        
        logging.info("Loading data...")

        # Load complete train and test sets
        self.train_path = f"train_{self.cfg.feature_version}.csv"
        self.test_path  = f"test_{self.cfg.feature_version}.csv"
        self.train = pd.read_csv(self.cfg.feature_store / self.train_path, parse_dates=["timestamp"], index_col="timestamp")
        self.test  = pd.read_csv(self.cfg.feature_store / self.test_path, parse_dates=["timestamp"], index_col="timestamp")

        logger.info("Data loaded.")


    def _training_study(self, model_name:str, experiment_id:str):

        # Get number of previous runs for this model
        model_runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.model_family='{model_name}'",
        )
        next_run_number = len(model_runs) + 1
        run_name = f"{model_name} run {next_run_number}"
        
        # Initiate the parent run and call the hyperparameter tuning child run logic
        with mlflow.start_run(
            experiment_id=experiment_id, 
            run_name=run_name, 
            nested=True,
            description=f"Parent run {next_run_number} of {model_name} in experiment {experiment_id}"
        ) as parent_run:
            
            parent_run_id = parent_run.info.run_id

            # Initialize the Optuna study
            study = optuna.create_study(direction="maximize")

            # Execute the hyperparameter optimization trials.
            # Note the addition of the `champion_callback` inclusion to control our logging
            study.optimize(
                lambda trial: objective(
                    trial=trial,
                    train=self.train,
                    target=self.cfg.target,
                    model_name=model_name,
                    search_space=self.cfg.search_spaces[model_name],
                    n_splits=self.cfg.splits,
                    experiment_id=experiment_id,
                    parent_run_id=parent_run_id,
                    seed=self.cfg.seed,
                ),
                n_trials=self.cfg.trials, 
                callbacks=[champion_callback]
            )

            # Log key data, artifacts and metadata of MLFlow parent run
            parent_logging(
                study=study,
                model_name=model_name,
                feature_version=self.cfg.feature_version,
                train=self.train,
                test=self.test,
                target=self.cfg.target,
                train_path=self.train_path,
                test_path=self.test_path,
                config_dir=self.cfg.configs,
                run_num=next_run_number,
            )

            # # Get the logged model uri so that we can load it from the artifact store
            # model_uri = mlflow.get_artifact_uri(artifact_path)


    def _loop_studies(self):

        experiment_id = get_or_create_experiment(self.cfg.experiment_name)

        self._load_data()

        optuna.logging.set_verbosity(self.cfg.verbosity)

        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        
        for model_name in self.cfg.models_to_run:

            self._training_study(
                model_name=model_name,
                experiment_id=experiment_id,
            )


    def run(self):
        self._loop_studies()
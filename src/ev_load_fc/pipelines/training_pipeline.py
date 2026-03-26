import pandas as pd
import pathlib
import mlflow
import optuna
from dataclasses import dataclass
from ev_load_fc.training.mlflow_api import init_mlflow, backup_mlflow_db, get_or_create_experiment, parent_logging
from ev_load_fc.training.optuna_api import objective, champion_callback
import logging
logger = logging.getLogger(__name__)


@dataclass
class TrainingPipelineConfig:
    # Paths and data
    feature_store: pathlib.Path
    configs: pathlib.Path
    images: pathlib.Path
    train: str
    test: str
    # MLFlow parameters
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
        self.train_path = self.cfg.feature_store/f"{self.cfg.train}_{self.cfg.feature_version}.csv"
        self.test_path  = self.cfg.feature_store/f"{self.cfg.test}_{self.cfg.feature_version}.csv"
        self.train = pd.read_csv(self.train_path, parse_dates=["timestamp"], index_col="timestamp")
        self.test  = pd.read_csv(self.test_path, parse_dates=["timestamp"], index_col="timestamp")

        # Ensure integer columns are float64 for MLflow logging
        for col in self.train.select_dtypes(include='int64').columns:
            self.train[col] = self.train[col].astype('float64')
        for col in self.test.select_dtypes(include='int64').columns:
            self.test[col] = self.test[col].astype('float64')

        logger.info("Data loaded.")


    def _training_study(self, model_name:str, experiment_id:str):

        # Get number of previous runs for this model
        model_runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.model_family='{model_name}' AND tags.level='parent'",
        )
        if len(model_runs) > 0:
            last_run_name = model_runs['tags.mlflow.runName'][0]
            last_run_number = last_run_name[-1]
            next_run_number = int(last_run_number) + 1
        else:
            next_run_number = 1
        run_name = f"{model_name} run {next_run_number}"

        logger.info(f"Beginning MLFlow run with metric {self.cfg.metric}: {run_name}")

        # Initiate the parent run and call the hyperparameter tuning child run logic
        with mlflow.start_run(
            experiment_id=experiment_id, 
            run_name=run_name, 
            nested=True,
            description=f"Parent run {next_run_number} of {model_name} in experiment {experiment_id}",
        ) as parent_run:

            # Log tags
            mlflow.set_tags(
                tags={
                    "project": "EV Load Forecasting",
                    "optimizer_engine": "optuna",
                    "model_family": model_name,
                    "feature_set_version": self.cfg.feature_version,
                    "level": "parent",
                }
            )
            
            parent_run_id = parent_run.info.run_id

            # Initialize the Optuna study
            study = optuna.create_study(direction="minimize")

            try:

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
                        metric=self.cfg.metric,
                        experiment_id=experiment_id,
                        parent_run_name=run_name,
                        parent_run_id=parent_run_id,
                        seed=None,
                    ),
                    n_trials=self.cfg.trials, 
                    callbacks=[champion_callback]
                )
            finally:
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
                    images_dir=self.cfg.images,
                    run_num=next_run_number,
                    metric=self.cfg.metric,
                )
            

    def run(self):

        init_mlflow()

        backup_mlflow_db()

        experiment_id = get_or_create_experiment(self.cfg.experiment_name)

        self._load_data()

        optuna.logging.set_verbosity(self.cfg.verbosity)
        
        for model_name in self.cfg.models_to_run:

            self._training_study(
                model_name=model_name,
                experiment_id=experiment_id,
            )
import pandas as pd
import numpy as np
import mlflow
from optuna.trial import Trial, FrozenTrial
from optuna.study import Study
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from pandas.tseries.holiday import USFederalHolidayCalendar as calender
from ev_load_fc.training.registry import build_model
from ev_load_fc.training.mlflow_api import _log_model_flavour
from ev_load_fc.training.prophet_api import prophet_df_format, cv_score_prophet_model
import logging
logger = logging.getLogger(__name__)


def objective(
        trial:Trial,
        train:pd.DataFrame,
        target:str,
        model_name:str,
        search_space:dict,
        n_splits:int,
        metric:str="rmse",
        experiment_id:str|None=None,
        parent_run_name:str|None=None,
        parent_run_id:str|None=None,
        feature_set_version:str|None=None,
        seed:int=42,
    )->float:
    """Conducts a single trial of hyperparameter optimization using Optuna for the specified model and search space.

    Args:
        trial (Any): Optuna trial object.
        train (pd.DataFrame): Training dataset.
        target (str): Name of the target variable column in the training dataset.
        model_name (str): Model name.
        search_space (dict): Optuna search space.
        n_splits (int): Number of forecast windows for time series cross validation.
        seed (int, optional): Random seed for reproducibility

    Returns:
        float: Mean cross-validated score for the model with the given hyperparameters
    """
    
    with mlflow.start_run(
        nested=True, 
        experiment_id=experiment_id, 
        run_name=f"{parent_run_name} - trial {trial.number}",
        parent_run_id=parent_run_id,
        description=f"Child run {trial.number} for {parent_run_id}",
        ) as child_run:

        mlflow.set_tags(
            tags={
                "project": "EV Load Forecasting",
                "optimizer_engine": "optuna",
                "model_family": model_name,
                "feature_set_version": feature_set_version or "",
                "level": "child",
            }
        )

        params = {}
        # Populate params dict from search_space
        for param, range in search_space.items():
            if param == 'xgboost_dart_mode':
                pass
            range = sorted(search_space[param])
            # Bools ranges
            if isinstance(range[0], bool) and isinstance(range[-1], bool):
                params[param] = trial.suggest_categorical(param, choices=range)
            # Integer ranges
            elif isinstance(range[0], int) and isinstance(range[-1], int):
                params[param] = trial.suggest_int(param, range[0], range[-1])
            # Float ranges
            elif isinstance(range[0], float) or isinstance(range[-1], float):
                params[param] = trial.suggest_float(param, range[0], range[-1])
            # Categorical values
            elif (not isinstance(range[0], float) and not isinstance(range[0], int)) or (not isinstance(range[-1], float) and not isinstance(range[-1], int)):
                params[param] = trial.suggest_categorical(param, choices=range)
            
        if model_name == "Prophet":
            cal = calender()
            holidays = cal.holidays(start=train.index.min(), end=train.index.max(), return_name=True)
            holidays_df = holidays.reset_index().rename(columns={'index':'ds', 0:'holiday'})
        else:
            holidays_df = None

        # Select the estimator
        est = build_model(model_name=model_name, params=params, holidays_df=holidays_df)
        
        X = train.drop(columns=[target])
        y = train[target]

        # Train and score model using cross validation
        if model_name == 'Prophet':
            scores = cv_score_prophet_model(
                model=est,
                y=y,
                n_splits=n_splits
            )
            mean_rmse = scores["rmse"]
            mean_mae = scores["mae"]
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            if isinstance(y, pd.Series):
                y = np.ravel(y)

            # Get both RMSE and MAE across Time Series Splits in one pass
            scores = cross_validate(
                estimator=est,
                X=X,
                y=y,
                cv=tscv,
                scoring={
                    "neg_rmse": "neg_root_mean_squared_error",
                    "neg_mae": "neg_mean_absolute_error",
                },
            )
            mean_rmse = -scores["test_neg_rmse"].mean()
            mean_mae = -scores["test_neg_mae"].mean()

        # Store both metrics on the trial so parent_logging can always access them
        trial.set_user_attr("rmse", mean_rmse)
        trial.set_user_attr("mae", mean_mae)

        # Log child run to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", mean_rmse)
        mlflow.log_metric("mae", mean_mae)

        # Refit on full training data and persist model artifact
        est_final = build_model(model_name=model_name, params=params, holidays_df=holidays_df)
        if model_name == "Prophet":
            est_final.fit(prophet_df_format(y))
        else:
            est_final.fit(X, y)
        flavour = _log_model_flavour(model_name)
        flavour.log_model(est_final, name="model")

    return mean_rmse if metric == "rmse" else mean_mae


def champion_callback(study:Study, frozen_trial:FrozenTrial):
    # Reference: https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Args:
        study (optuna.study.Study): The Optuna study object.
        frozen_trial (optuna.trial.FrozenTrial): The trial that was just completed.
    """
    
    # Extract current winner of study
    winner = study.user_attrs.get("winner", None)

    # If the current FrozenTrial value exceeds winner's, then update winner and log the achieved value (and improvement)
    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            logger.info(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value:.4f} with "
                f"{abs(improvement_percent):.4f}% improvement over previous winner"
            )
        else:
            logger.info(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value:.4f}")
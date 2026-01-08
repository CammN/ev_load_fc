import pandas as pd
import math
import numpy as np
import mlflow
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from prophet.diagnostics import cross_validation
from pandas.tseries.holiday import USFederalHolidayCalendar as calender
from ev_load_fc.training.registry import build_model
import logging
logger = logging.getLogger(__name__)


def prophet_df_format(ts:pd.Series) -> pd.DataFrame:
    """
    Formats a time series for use with Prophet model.

    Args:
        ts (pd.Series): Pandas Series to format

    Returns:
        pd.DataFrame: Formatted dataframe for Prophet model
    """
    # Copy original series and extract index and target names
    proph_df = ts.copy()
    index_name  =  proph_df.index.name
    target_name = proph_df.columns[0]

    proph_df = proph_df.reset_index()
    proph_df = proph_df.rename(columns={index_name:'ds', target_name:'y'})

    return proph_df[['ds','y']]


def cv_score_prophet_model(model, y:pd.Series, n_splits:int) -> float:
    """
    Conducts cross-validation for a Prophet model and returns the mean Root Mean Squared Error across splits.

    Args:
        model (Prophet): A Prophet model instance.
        y (any): Input time series data.
        n_splits (int): Number of CV splits.

    Raises:
        ValueError: If y is not in correct format and target is not provided.

    Returns:
        float: Mean Root Mean Squared Error across CV splits.
    """

    # Ensure y is in correct format for Prophet modelling
    if 'ds' not in y.columns or 'y' not in y.columns:
        y_proph = prophet_df_format(y)
    else:
        y_proph = y[['ds','y']].copy()

    # Fit model and cross validate
    model.fit(y_proph)
    window_length = math.floor((len(y_proph)-366)/n_splits)
    cv = cross_validation(model=model, initial='366 days', horizon=f'{window_length} days', period=f'{window_length} days', disable_tqdm=True)

    # Calculate Root Mean Squared Error for each CV split
    split_rsmes = {}
    for cutoff in cv['cutoff'].unique():
        split = cv[cv['cutoff']==cutoff]
        min_date, max_date = split['ds'].min(), split['ds'].max()
        actual = y_proph[y_proph['ds'].between(min_date, max_date, inclusive='both')]
        rsme = root_mean_squared_error(actual['y'], split['yhat'])
        split_rsmes[cutoff] = rsme

    # Take mean RSME
    mean_rsme = np.mean(list(split_rsmes.values()))

    return mean_rsme


def objective(
        trial, 
        train:pd.DataFrame, 
        target:str, 
        model_name:str, 
        search_space:dict, 
        n_splits:int, 
        experiment_id:str|None=None,
        parent_run_id:str|None=None,
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
        parent_run_id=parent_run_id,
        description=f"Child run {trial.number} for {parent_run_id}",
        ) as child_run:

        params = {}
        # Populate params dict from search_space
        for param, range in search_space.items():
            range = sorted(search_space[param])
            # Integer ranges
            if isinstance(range[0], int) and isinstance(range[-1], int):
                params[param] = trial.suggest_int(param, range[0], range[-1])
            # Float ranges
            elif isinstance(range[0], float) or isinstance(range[-1], float):
                params[param] = trial.suggest_float(param, range[0], range[-1])
            # Categorical values
            elif (not isinstance(range[0], float) and not isinstance(range[0], int)) or (not isinstance(range[-1], float) and not isinstance(range[-1], int)):
                params[param] = trial.suggest_categorical(param, choices=range)
        trial.suggest_categorical("random_state", choices=[seed])
            
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
            mean_score = cv_score_prophet_model(
                model=est, 
                y=y, 
                n_splits=n_splits
            )
        else:
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            if isinstance(y, pd.Series):
                y = np.ravel(y)

            # Get scores across Time Series Splits
            scores = cross_val_score(
                estimator=est,
                X=X, 
                y=y,
                cv=tscv,
                scoring='neg_root_mean_squared_error'
            )
            mean_score = scores.mean()

        # Log child run to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", mean_score)
        mlflow.set_tags(
            tags={
                "project": "EV Load Forecasting",
                "optimizer_engine": "optuna",
                "model_family": model_name,
            }
        )

    return mean_score

def champion_callback(study, frozen_trial):
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
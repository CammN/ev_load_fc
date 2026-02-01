import mlflow
import pandas as pd
from pathlib import Path
from optuna.study import Study
from optuna.visualization import plot_param_importances, plot_optimization_history
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.sources import LocalArtifactDatasetSource
from pandas.tseries.holiday import USFederalHolidayCalendar as calender
from ev_load_fc.training.registry import build_model 
from ev_load_fc.training.evaluation import EvaluationPlots


def get_or_create_experiment(experiment_name:str, tags:dict={}) -> str :
    # Reference: https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/notebooks/hyperparameter-tuning-with-child-runs/
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.
    - tags (dict): Optional set of tags to assign to MLFlow experiment upon creation.
    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name, tags=tags)



def parent_logging(
    study:Study,
    model_name:str,
    feature_version:str,
    train:pd.DataFrame,
    test:pd.DataFrame,
    target:str,
    train_path:Path,
    test_path:Path,
    config_dir:Path,
    images_dir:Path,
    run_num:int,
) -> None:
    """
    Logs the best model and evaluation plots to MLflow as a parent run.
    Args:
        study (optuna.study.Study): The Optuna study object containing the optimization results.
        model_name (str): The name of the model family.
        feature_version (str): The version of the feature set used.
        train (mlflow.data.pandas_dataset.PandasDataset): The training dataset.
        test (mlflow.data.pandas_dataset.PandasDataset): The testing dataset.
        target (str): The name of the target variable.
        train_path (pathlib.Path): The file path to the training dataset.
        test_path (pathlib.Path): The file path to the testing dataset.
        config_dir (pathlib.Path): The directory containing configuration files.
        run_num (int): The run number for logging purposes.
    Returns:
        None
    """

    # Log best model's parameters and score
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_rmse", study.best_value)

    # Create MLFlow datasets
    train_source = LocalArtifactDatasetSource(str(train_path))
    test_source = LocalArtifactDatasetSource(str(test_path))
    train_log = mlflow.data.from_pandas(
        train,
        source=train_source,
        name=f"ev_fc_train-{feature_version}",
        targets=target,
    )
    test_log = mlflow.data.from_pandas(
        test,
        source=test_source,
        name=f"ev_fc_test-{feature_version}",
        targets=target,
    )


    # Log input data
    mlflow.log_input(train_log, context="training", tags={"feature_version":feature_version})
    mlflow.log_input(test_log, context="testing", tags={"feature_version":feature_version})

    # Log config.yaml
    mlflow.log_artifact(local_path=str(config_dir/"config.yaml"), artifact_path="config")

    if model_name == "Prophet":
        cal = calender()
        holidays = cal.holidays(start=train.index.min(), end=train.index.max(), return_name=True)
        holidays_df = holidays.reset_index().rename(columns={'index':'ds', 0:'holiday'})
    else:
        holidays_df = None

    # Split out input and target features
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test  = test.drop(columns=[target])
    y_test  = test[target]

    # Lazily import best model
    model = build_model(
        model_name=model_name,
        params=study.best_params,
        holidays_df=holidays_df,
    )
    # Fit
    model = model.fit(X_train, y_train)

    ### Create and log plots
    plotter = EvaluationPlots(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model=model,
        model_name=model_name,
        run_num=run_num,
    )
    # Log the correlation plot
    correlation_plot = plotter.plot_correlation_with_target()
    corr_path = str(images_dir/"plots"/f"{model_name}_{run_num}_correlations.png")
    mlflow.log_figure(figure=correlation_plot, artifact_file=corr_path)
    # Log the feature importances plot
    importances = plotter.plot_feature_importance()
    feat_imp_path = str(images_dir/"plots"/f"{model_name}_{run_num}_feature_importances.png")
    mlflow.log_figure(figure=importances, artifact_file=feat_imp_path)
    # Log the residuals plot
    residuals = plotter.plot_residuals()
    resid_path = str(images_dir/"plots"/f"{model_name}_{run_num}_residuals.png")
    mlflow.log_figure(figure=residuals, artifact_file=resid_path)
    # Optuna study plots
    fig_param_importance = plot_param_importances(study)
    fig_optimization_history = plot_optimization_history(study)
    param_imp_path = str(images_dir/"plots"/f"{model_name}_{run_num}_param_importances.html")
    opt_hist_path = str(images_dir/"plots"/f"{model_name}_{run_num}_optimization_history.html")
    mlflow.log_figure(fig_param_importance,artifact_file=param_imp_path)
    mlflow.log_figure(fig_optimization_history,artifact_file=opt_hist_path)

    
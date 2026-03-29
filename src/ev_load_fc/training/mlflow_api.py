import mlflow
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path
from optuna.study import Study
from optuna.visualization import plot_param_importances, plot_optimization_history
from mlflow.data.sources import LocalArtifactDatasetSource
from pandas.tseries.holiday import USFederalHolidayCalendar as calender
from ev_load_fc.training.registry import build_model 
from ev_load_fc.training.evaluation import EvaluationPlots
from ev_load_fc.config import PROJECT_ROOT, CFG
from types import ModuleType

# All MLflow data lives in one explicit, versioned location
MLFLOW_DIR = Path(f"{PROJECT_ROOT}/mlflow_store")
MLFLOW_DB  = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"


def init_mlflow():
    """
    Safely initialise MLflow tracking. 
    Will never create a new DB if one already exists at the configured path.
    """
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)

    # Sanity check - print URI so you can always see where you're pointed
    print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}")
    print(f"DB exists           : {MLFLOW_DB.exists()}")

    return mlflow.get_tracking_uri()


def backup_mlflow_db():
    """Backs up all MLflow runs to be safe when calling long training run."""

    if not MLFLOW_DB.exists():
        print("No MLflow DB found yet — skipping backup (no runs logged yet).")
        return

    backup_dir = MLFLOW_DIR / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"mlflow_{timestamp}.db"
    shutil.copy2(MLFLOW_DB, dest)
    print(f"Backed up to {dest}")

    # Keep only last 10 backups
    backups = sorted(backup_dir.glob("mlflow_*.db"))
    for old in backups[:-10]:
        old.unlink()
        print(f"Removed old backup: {old.name}")
        

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


def _log_model_flavour(model_name: str) -> ModuleType:
    """Logs a model to an active MLflow run using the correct flavour."""
    flavour_map = {
        "XGBoost"       : mlflow.xgboost,
        "LightGBM"      : mlflow.lightgbm,
        "CatBoost"      : mlflow.catboost,
        "Prophet"       : mlflow.prophet,
        "Random Forest" : mlflow.sklearn,
        "AdaBoost"      : mlflow.sklearn,
    }
    flavour = flavour_map.get(model_name, mlflow.sklearn)
    
    return flavour    


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
    metric:str="rmse",
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
    mlflow.log_metric(f"best_{metric}", study.best_value)

    # Create MLFlow datasets
    train_name = CFG['files']['train']
    test_name  = CFG['files']['test']
    train_source = LocalArtifactDatasetSource(str(train_path))
    test_source = LocalArtifactDatasetSource(str(test_path))
    train_log = mlflow.data.from_pandas(
        train,
        source=train_source,
        name=f"ev_fc_{train_name}-{feature_version}",
        targets=target,
    )
    test_log = mlflow.data.from_pandas(
        test,
        source=test_source,
        name=f"ev_fc_{test_name}-{feature_version}",
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
    # Log best fitted model
    model_flavour = _log_model_flavour(model_name)
    model_flavour.log_model(model, artifact_path='model')

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

    
def get_best_model(experiment_name:str, metric:str = "rmse", ascending:bool = False, filter_string:str|None=None):
    """
    Finds and returns the best model from an MLflow experiment based on a given metric.

    Args:
        experiment_name (str): Name of the MLflow experiment to search.
        metric (str): The metric to rank runs by. Defaults to 'val_rmse'.
        ascending (bool): If True, lower metric is better (e.g. RMSE). If False, higher is better (e.g. R2). Defaults to True.

    Returns:
        tuple: (model, run) where model is the loaded sklearn-compatible model and run is the best MLflow RunInfo object.

    Raises:
        ValueError: If the experiment is not found or no successful runs exist.
    """

    tracking_uri = init_mlflow()
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    full_string = 'status = "FINISHED"'
    if filter_string:
        full_string += ' AND ' + filter_string

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=full_string,
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
    )

    if runs.empty:
        raise ValueError(f"No finished runs found in experiment '{experiment_name}'.")
    if f"metrics.{metric}" not in runs.columns or runs[f"metrics.{metric}"].isna().all():
        raise ValueError(f"Metric '{metric}' not found in any runs.")

    best_run = runs.iloc[0]
    best_run_id = best_run["run_id"]
    run_name = best_run.get("tags.mlflow.runName", "")

    flavour = _log_model_flavour(next(
        (name for name in ["XGBoost", "LightGBM", "CatBoost", "Prophet", "Random Forest", "AdaBoost"]
         if name.lower() in run_name.lower()),
        "sklearn"
    ))
    local_path = mlflow.artifacts.download_artifacts(f"runs:/{best_run_id}/model")
    model = flavour.load_model(Path(local_path).as_uri())

    print(f"Best run ID : {best_run_id}")
    print(f"Best {metric}  : {best_run[f'metrics.{metric}']:.4f}")
    print(f"Model        : {best_run.get('tags.mlflow.runName', 'N/A')}")

    return model, best_run

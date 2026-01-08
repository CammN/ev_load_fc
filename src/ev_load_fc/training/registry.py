# This module allows to lazily load necessary model objects
import pandas as pd

def get_random_forest():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor

def get_adaboost():
    from sklearn.ensemble import AdaBoostRegressor
    return AdaBoostRegressor

def get_xgboost():
    from xgboost import XGBRegressor
    return XGBRegressor

def get_lightgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor

def get_catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor

def get_prophet():
    from prophet import Prophet
    return Prophet


MODEL_REGISTRY = {
    "Random Forest": get_random_forest,
    "AdaBoost": get_adaboost,
    "XGBoost": get_xgboost,
    "LightGBM": get_lightgbm,
    "CatBoost": get_catboost,
    "Prophet": get_prophet,
}


def build_model(model_name: str, params: dict, holidays_df:pd.DataFrame=None):
    """ 
    Lazy model builder that initializes and returns model instances based on the model name.

    Args:
        model_name (str): Name of the model to build
        params (dict): Hyperparameters for the model
        holidays_df (pd.DataFrame, optional): DataFrame containing information on holidays for use in Prophet. Defaults to None.
            - One column 'ds' with datetime values
            - One column 'holiday' with holiday names

    Raises:
        ValueError: If the model_name is not found in the MODEL_REGISTRY.

    Returns:
        (Any): An instance of the specified model initialized with the given parameters.
    """
    try:
        model_factory = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}")

    model_cls = model_factory()
    if holidays_df is not None:
        return model_cls(holidays=holidays_df, **params)
    return model_cls(**params)
# This module allows to lazily load necessary model objects
import pandas as pd

def get_random_forest():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor

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
    "XGBoost": get_xgboost,
    "LightGBM": get_lightgbm,
    "CatBoost": get_catboost,
    "Prophet": get_prophet,
}


def build_model(model_name:str, params:dict, holidays_df:pd.DataFrame|None=None):
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
        (Any): An instance of the specified
    """
    params = params.copy()

    try:
        model_factory = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}")
    model_cls = model_factory()

    # Prophet requires holidays to be passed during initialization
    if holidays_df is not None:
        return model_cls(holidays=holidays_df, **params)

    # XGBoost dart-specific params should be removed if booster != dart so they are not suggested in Optuna studies
    if model_name == "XGBoost" and params.get("booster") != "dart":
        for dart_param in ["sample_type", "normalize_type", "rate_drop", "skip_drop"]:
            params.pop(dart_param, None)
    # XGBoost tree-specific params should be removed if booster == gblinear so they are not suggested in Optuna studies
    if model_name == "XGBoost" and params.get("booster") == "gblinear":
        for tree_param in ["colsample_bytree", "max_depth", "max_leaves", "min_child_weight", "min_split_loss", "subsample", "tree_method"]:
            params.pop(tree_param, None)

    # LightGBM dart-specific params should be removed if boosting_type != dart.
    if model_name == "LightGBM" and params.get("boosting_type") != "dart":
        for dart_param in ["drop_rate", "skip_drop", "xgboost_dart_mode", "uniform_drop"]:
            params.pop(dart_param, None)

     # CatBoost conditional parameter logic
    if model_name == "CatBoost":
        bootstrap_type = params.get("bootstrap_type")
        grow_policy = params.get("grow_policy")
        boosting_type = params.get("boosting_type")
        leaf_estimation_method = params.get("leaf_estimation_method")

        # --- Bootstrap-related ---
        # bagging_temperature is only valid for Bayesian bootstrap
        if bootstrap_type != "Bayesian":
            params.pop("bagging_temperature", None)

        # subsample is only valid for Bernoulli or MVS bootstrap
        if bootstrap_type not in ("Bernoulli", "MVS"):
            params.pop("subsample", None)

        # mvs_reg is only valid for MVS bootstrap
        if bootstrap_type != "MVS":
            params.pop("mvs_reg", None)

        # --- Grow policy-related ---
        # max_leaves is only valid for Lossguide grow policy
        if grow_policy != "Lossguide":
            params.pop("max_leaves", None)

        # min_data_in_leaf is only valid for Lossguide or Depthwise
        if grow_policy == "SymmetricTree":
            params.pop("min_data_in_leaf", None)

        # --- Ordered boosting conflicts ---
        # Ordered boosting is only supported for SymmetricTree - force it if Ordered is set
        if boosting_type == "Ordered" and grow_policy != "SymmetricTree":
            params.pop("grow_policy", None)  # let CatBoost default to SymmetricTree
            params.pop("max_leaves", None)   # clean up any Lossguide params that snuck in
            params.pop("min_data_in_leaf", None)

        # Ordered boosting specific params - remove if not Ordered
        if boosting_type != "Ordered":
            for ordered_param in ["fold_len_multiplier", "fold_permutation_block"]:
                params.pop(ordered_param, None)

        # leaf_estimation_method=Exact is not supported with Ordered boosting
        if boosting_type == "Ordered" and leaf_estimation_method == "Exact":
            params.pop("leaf_estimation_method", None)

        # --- Leaf estimation-related ---
        # Newton-based score functions require Newton leaf estimation
        if params.get("score_function") in ("NewtonCosine", "NewtonL2"):
            params["leaf_estimation_method"] = "Newton"

        # leaf_estimation_iterations > 1 is meaningless for Exact method
        if leaf_estimation_method == "Exact":
            params.pop("leaf_estimation_iterations", None)

    return model_cls(**params)
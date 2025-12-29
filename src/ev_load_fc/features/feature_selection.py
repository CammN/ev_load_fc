import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, RFE, RFECV, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import logging
logger = logging.getLogger(__name__)


def k_by_scores(X:pd.DataFrame, y:pd.Series, method:str, k:int, seed:int=None) -> pd.DataFrame:
    """Runs feature selection for a set of exogenous features using a given method to produce the k best features.

    Args:
        X (pd.DataFrame): Exogenous feature DataFrame
        y (pd.Series): Target feature Series
        method (str): Denotes the chosen method for feature selection 
                        - 'f' for univariate feature selection using the F-statistic
                        - 'mi' for univariate feature selection using mutual information score
                        - 'imp' for impurity based feature selection using a RandomForestRegressor
                        - 'rfe' for recursive feature selection using a RandomForestRegressor
                        - 'rfecv' for recursive feature selection with TimeSeriesSplit cross validation using a RandomForestRegressor
        k (int): Number of top features to select from X
        seed (int, optional): Random seed for reproducible feature selection when using the 'imp' or 'rfe' methods

    Raises:
        ValueError: Wwhen k is greater than the number of features in X
        ValueError: Wwhen an invalid method is given

    Returns:
        pd.DataFrame: Top k exogenous features as a DataFrame
    """

    X_cut = X.copy()
    y_arr = np.ravel(y)
    dims = X_cut.shape
    logger.debug(f"Exogenous feature set has dimensions {dims}")

    if k > dims[1]:
        logger.error(f"Invalid k: {k} > {dims[1]}")
        raise ValueError(
            f"k must be <= number of features ({dims[1]}), got {k}"
        )

    if method == 'f':
        logger.debug(f"Performing univariate feature selection using the F-statistic for {k} features")
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X_cut, y_arr)
        X_cut = pd.DataFrame(
            X_selected,
            columns=X_cut.columns[selector.get_support()],
            index=X_cut.index
        )
        logger.debug(f"Selected {k} best features with respect to F-statistic")

    elif method == 'mi':
        logger.debug(f"Performing univariate feature selection using mutual information score for {k} features")
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_selected = selector.fit_transform(X_cut, y_arr)
        X_cut = pd.DataFrame(
            X_cut,
            columns=X_cut.columns[selector.get_support()],
            index=X_cut.index
        )
        logger.debug(f"Selected {k} best features with respect to mutual information score")

    elif method == 'imp':
        logger.debug(f"Performing feature selection using Random Forest impurity for {k} features")
        selector = RandomForestRegressor(random_state=seed, verbose=1, n_jobs=-1)
        selector.fit(X_cut, y_arr)
        impurities = pd.Series(selector.feature_importances_, index=X_cut.columns).sort_values(ascending=False)
        top_k_cols = impurities.iloc[:k].index
        X_cut = X_cut[top_k_cols]
        logger.debug(f"Selected {k} best features with respect to impurities of Random Forest model")

    elif method == "rfe":
        logger.debug(f"Performing recursive feature elimination using a Random Forest for {k} features")
        selector = RFE(RandomForestRegressor(random_state=seed, n_jobs=-1), 
                       n_features_to_select=k, 
                       step=1, 
                       verbose=1)
        X_selected = selector.fit_transform(X_cut, y_arr)
        X_cut = pd.DataFrame(
            X_selected,
            columns=X_cut.columns[selector.get_support()],
            index=X_cut.index
        )
        logger.debug(f"Selected {k} best features using recursive feature elimination with a Random Forest model")

    elif method == "rfecv":
        logger.debug(f"Performing recursive feature elimination with cross validation using a Random Forest for {k} features")
        selector = RFECV(RandomForestRegressor(random_state=seed, n_jobs=-1), 
                         min_features_to_select=k, 
                         step=3, 
                         cv=TimeSeriesSplit(n_splits=3), 
                         verbose=1)
        X_selected = selector.fit_transform(X_cut, np.ravel(y))
        X_cut = pd.DataFrame(
            X_selected,
            columns=X_cut.columns[selector.get_support()],
            index=X_cut.index
        )
        logger.debug(f"Selected {len(X_cut.columns)} best features using recursive feature elimination with cross validation and a Random Forest model")

    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    return X_cut


def remove_correlated_features(X, y):
    pass
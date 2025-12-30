import pandas as pd
import numpy as np
import pathlib
from dataclasses import dataclass 
from ev_load_fc.data.preprocessing import scale_features
from ev_load_fc.features.feature_creation import (
    aggregate_features, 
    time_features, 
    ohe_holidays,
    lag_features, 
    rolling_window_features,
    flatten_nested_dict,
)
from ev_load_fc.features.feature_selection import k_by_scores
import logging
logger = logging.getLogger(__name__)


@dataclass
class FeaturePipelineConfig:
    # Paths
    combined_path: pathlib.Path
    feature_store: pathlib.Path
    # Miscellaneous
    min_timestamp: pd.Timestamp
    max_timestamp: pd.Timestamp
    split_date: pd.Timestamp
    seed: int
    # Feature creation parameters
    target: str
    holiday_list: list
    time_feature_dict: dict
    energy_col_substrs: list
    weather_col_substrs: list
    temperature_col_substrs: list
    traffic_col_substrs: list
    # Feature selection parameters
    scale_method: str
    k_1: int
    fe_method_1: str
    k_2: int
    fe_method_2: str
    corr_thresh: float
    holidays: set
    # Optional runtime parameters
    run_feat_eng: bool
    run_feat_select: bool


class FeaturePipeline:


    def __init__(self, config: FeaturePipelineConfig):
        self.cfg = config


    def _feature_engineering(self):

        raw_combined = pd.read_csv(
            self.cfg.combined_path, 
            parse_dates=['timestamp'], 
            index_col=['timestamp'], 
            low_memory=False
        )
        combined = raw_combined.copy()
        col_count_1 = combined.shape[1]
        logger.debug(f"Number of columns in combined dataset: {col_count_1}")

        ## 1 Aggegate features ##

        # Aggregate weather event duration by weather event type
        combined = aggregate_features(df=combined, out_name='rain_dur', substr1='rain')
        combined = aggregate_features(df=combined, out_name='fog_dur', substr1='fog')
        # Aggregate traffic event duration and distance by traffic event type 
        combined = aggregate_features(df=combined, out_name='cong_dis', substr1='distance_congestion')
        combined = aggregate_features(df=combined, out_name='flow_dis', substr1='distance_flow')
        combined = aggregate_features(df=combined, out_name='cong_dur', substr1='duration_congestion')
        combined = aggregate_features(df=combined, out_name='flow_dur', substr1='duration_flow')

        col_count_2 = combined.shape[1]
        logger.debug(f"Number of column aggregations: {col_count_2-col_count_1}")

        # Identify features from each data source
        energy_cols = [col for col in combined.columns if any([s in col for s in self.cfg.energy_col_substrs])]
        weather_cols = [col for col in combined.columns if any([s in col for s in self.cfg.weather_col_substrs])]
        temperature_cols = [col for col in combined.columns if any([s in col for s in self.cfg.temperature_col_substrs])]
        traffic_cols = [col for col in combined.columns if any([s in col for s in self.cfg.traffic_col_substrs])]

        ## 2 Date features ##

        combined = time_features(combined)

        holidays = ohe_holidays(self.cfg.holiday_list, self.cfg.min_timestamp, self.cfg.max_timestamp)
        combined = combined.merge(holidays, how='left', left_index=True, right_index=True)

        col_count_3 = combined.shape[1]
        logger.debug(f"Number of date and holiday features created: {col_count_3-col_count_2}")

        ## 3 Lag features ##
        lag_dict = {}
        lags = self.cfg.time_feature_dict['lags']

        if 'energy' in lags.keys():
            for energy_col in energy_cols:
                lag_dict[energy_col] = lags['energy']

        combined = lag_features(combined, lag_dict)

        col_count_4 = combined.shape[1]
        logger.debug(f"Number of lag features created: {col_count_4-col_count_3}")

        ## 4 Rolling sum features ##
        rw_sum_dict = {}
        rw_sums = self.cfg.time_feature_dict['rolling_sums']

        if 'energy' in rw_sums.keys():
            for energy_col in energy_cols:
                rw_sum_dict[energy_col] = rw_sums['energy']

        if 'weather' in rw_sums.keys():
            for weather_col in weather_cols:  
                rw_sum_dict[weather_col] = rw_sums['weather']

        if 'traffic' in rw_sums.keys():
            for traffic_col in traffic_cols:  
                rw_sum_dict[traffic_col] = rw_sums['traffic']

        combined = rolling_window_features(combined, rw_sum_dict, agg_func='sum')

        col_count_5 = combined.shape[1]
        logger.debug(f"Number of rolling sum features created: {col_count_5-col_count_4}")

        ## 5 Rolling mean features ##
        rw_mean_dict = {}
        rw_means = self.cfg.time_feature_dict['rolling_means']

        if 'energy' in rw_means.keys():
            for energy_col in energy_cols:
                rw_mean_dict[energy_col] = rw_means['energy']

        if 'temperature' in rw_means.keys():
            for temperature_col in temperature_cols:  
                rw_mean_dict[temperature_col] = rw_means['temperature']

        combined = rolling_window_features(combined, rw_mean_dict, agg_func='mean')

        col_count_6 = combined.shape[1]
        logger.debug(f"Number of rolling mean features created: {col_count_6-col_count_5}")

        ## 6 Split into input features and target feature(s) ##

        # Cutoff window size at start of time frame
        all_lags = flatten_nested_dict(self.cfg.time_feature_dict)
        max_lag  = max(all_lags)

        # Collect list of input features from combined df
        features = [
            col for col in combined.columns
            if '_sin' in col 
            or '_cos' in col
            or '_lag_' in col
            or '_rw_' in col
        ]
        features = features + self.cfg.holiday_list

        logger.debug(f"Number of input features created: {len(features)}")

        # Input & target sets
        self.X = combined.iloc[max_lag:][features].copy()
        self.y = combined.iloc[max_lag:][self.cfg.target].copy()
        logger.debug(f"Shape of input feature set X: {self.X.shape}")
        logger.debug(f"Shape of target feature set y: {self.y.shape}")
        # Save
        self.X.to_csv(self.cfg.feature_store / "X.csv", index_label="timestamp")
        self.y.to_csv(self.cfg.feature_store / "y.csv", index_label="timestamp")


    def _feature_selection(self):
        """ Perform feature selection on training and test sets."""    

        # Read in input and target features 
        X = pd.read_csv(
            self.cfg.feature_store / "X.csv", 
            parse_dates=['timestamp'], 
            index_col=['timestamp'], 
            low_memory=False
        )
        y = pd.read_csv(
            self.cfg.feature_store / "y.csv", 
            parse_dates=['timestamp'], 
            index_col=['timestamp'], 
            low_memory=False
        )

        # Scale features (optional)
        if len(self.cfg.scale_method) > 0:
            X = scale_features(X, self.cfg.scale_method)

        # Split into train and test
        X_train = X[X.index <  self.cfg.split_date].copy()
        y_train = y[y.index <  self.cfg.split_date].copy()
        X_test  = X[X.index >= self.cfg.split_date].copy()
        y_test  = y[y.index >= self.cfg.split_date].copy()

        # Select K best features using given method (first round)
        X_train_cut = k_by_scores(
            X=X_train, 
            y=y_train, 
            method=self.cfg.fe_method_1, 
            k=self.cfg.k_1,
            seed=self.cfg.seed
        )

        # Select K best features using given method (second round)
        X_train_cut = k_by_scores(
            X=X_train_cut, 
            y=y_train, 
            method=self.cfg.fe_method_2, 
            k=self.cfg.k_2,
            seed=self.cfg.seed
        )

        # Drop highly correlated features (optional)
        if self.cfg.corr_thresh < 1:
            pass

        ## Add back in certain features ##
        # If a one of a pair of time-based features is selected, ensure both are selected
        add_back = set()
        if any('hour_' in col for col in X_train_cut.columns):
            add_back.update({'hour_sin','hour_cos'})
        if any('weekday_' in col for col in X_train_cut.columns):
            add_back.update({'weekday__sin','weekday__cos'})
        if any('month_' in col for col in X_train_cut.columns):
            add_back.update({'month__sin','month__cos'})
        # Ensure all holiday features are included
        add_back.update(self.cfg.holidays)

        # Create final feature set
        final_features = set(X_train_cut.columns)
        final_features.update(add_back)
        X_train = X_train.loc[:, X_train.columns.intersection(final_features)]
        logger.debug(f"Added back {len(add_back)} features into feature set, ending with {len(final_features)} in total")

        # Match features of test set to that of train set
        X_test = X_test[X_train.columns].copy()
        # Save model data
        train = pd.concat([X_train, y_train], axis=1)
        test  = pd.concat([X_test, y_test], axis=1)

        version = f"{self.cfg.fe_method_1}_{self.cfg.k_1}_{self.cfg.fe_method_2}_{self.cfg.k_2}"

        train.to_csv(self.cfg.feature_store / f"train_{version}.csv", index=True, index_label='timestamp')
        test.to_csv(self.cfg.feature_store / f"test_{version}.csv", index=True, index_label='timestamp')
        logger.debug(f"Saved reduced train and tests sets to the {self.cfg.feature_store} directory")


    def run(self):
        if self.cfg.run_feat_eng:
            logger.info("Beginning _feature_engineering")
            self._feature_engineering()
            logger.info("Finished _feature_engineering")
        if self.cfg.run_feat_select:
            logger.info("Beginning _feature_selection")
            self._feature_selection()
            logger.info("Finished _feature_selection")
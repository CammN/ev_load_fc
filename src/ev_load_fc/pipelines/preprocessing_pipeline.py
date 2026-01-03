import pandas as pd
import numpy as np
import pathlib
from dataclasses import dataclass 
from ev_load_fc.data.preprocessing import (
    clean_enrich_split,
    mad_outlier_bounds, 
    cap_outliers_mad, 
    avg_temp_tracker, 
    rolling_mad_outliers,
    split_event_by_hour,
    validate_time_series
)
from ev_load_fc.features.feature_creation import aggregate_features
import logging
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingPipelineConfig:
    # Paths
    processed_data_path: pathlib.Path
    ev_int_path: pathlib.Path
    weather_int_path: pathlib.Path
    temp_path: pathlib.Path
    traffic_int_path: pathlib.Path
    ev_proc_path: pathlib.Path
    weather_proc_path: pathlib.Path
    temp_proc_path: pathlib.Path
    traffic_proc_path: pathlib.Path
    combined_path: pathlib.Path
    # Filters
    min_timestamp: pd.Timestamp
    max_timestamp: pd.Timestamp
    ev_cols: list
    weather_cols: list
    temp_cols: list
    traffic_cols: list
    weather_type_filt: list
    traffic_type_filt: list
    # Preprocessing parameters
    kw_quant: float
    mad_thresh: float
    split_date: pd.Timestamp
    sampling_interval: str
    min_outlier_samples: int
    # Optional runtime parameters
    run_ev: bool
    run_weather: bool
    run_temperature: bool
    run_traffic: bool
    run_combine: bool


class PreprocessingPipeline:
    """Pipeline to preprocess EV, weather (including temperature), and traffic datasets for feature engineering."""


    def __init__(self, config: PreprocessingPipelineConfig):
        self.cfg = config
        self.train_date_range = pd.date_range(
             start=self.cfg.min_timestamp,
             end=self.cfg.split_date-pd.to_timedelta(self.cfg.sampling_interval),
             freq=self.cfg.sampling_interval
        )  
        self.test_date_range  = pd.date_range(
            start=self.cfg.split_date,
            end=self.cfg.max_timestamp-pd.to_timedelta(self.cfg.sampling_interval),
            freq=self.cfg.sampling_interval
        )
        self.full_date_range  = pd.date_range(
            start=self.cfg.min_timestamp,
            end=self.cfg.max_timestamp-pd.to_timedelta(self.cfg.sampling_interval),
            freq=self.cfg.sampling_interval
        ) 


    def _process_ev(self):

        ### 1. Clean Data ###

        ev = pd.read_csv(self.cfg.ev_int_path, parse_dates=['starttime','endtime','transaction_date'], low_memory=False)

        # Duration of charging events in hours
        ev['duration_hours'] = pd.to_timedelta(ev['charging_time']).dt.total_seconds() / 3600

        # Approximate charger plug power
        ev['charger_kw'] = ev['energy'] / ev['duration_hours'] 
        # Drop rows with missing plug types
        pre_plug_na = len(ev)
        ev = ev[~(ev['plug_type'].isna())]
        logger.debug(f"Dropped {pre_plug_na-len(ev)} rows from EV dataset for missing plug type")

        # Function to drop duplicates/missings/anomalous rows etc, enrich and then split out train set for calculate anomalous boundaries
        ev, ev_train = clean_enrich_split(
            df=ev,
            df_name='EV',
            keep_cols=self.cfg.ev_cols,
            positive_cols=['duration_hours'],
            log_cols=['energy'],
            split_date=self.cfg.split_date
        )

        ### Remove extreme values based on calculated charger plug power by the x% upper and lower quantiles of the EV train set
        ev['kw_drop'] = 0
        for pt in ev_train['plug_type'].unique():
            # Calculate boundaries based on train set only
            low_quant = ev_train[ev_train['plug_type']==pt]['charger_kw'].quantile(self.cfg.kw_quant)
            upp_quant = ev_train[ev_train['plug_type']==pt]['charger_kw'].quantile(1-self.cfg.kw_quant)
            # Create filter across entire EV dataset
            ext_val_mask = ((low_quant >= ev['charger_kw']) | (ev['charger_kw'] >= upp_quant))
            ev.loc[(ev['plug_type']==pt) & ext_val_mask, 'kw_drop'] = 1
        pre_kw_filt = len(ev)
        ev = ev[ev['kw_drop']!=1]
        logger.debug(f"Dropped {pre_kw_filt-len(ev)} rows from EV dataset for having anomalous charging power")

        ### Impute missing values for energy using approximate power for charger plug type 
        j_plug_cond = (ev['energy'].isna()) & (ev['plug_type']=='J1772')
        n_plug_cond = (ev['energy'].isna()) & (ev['plug_type']=='NEMA 5-20R')
        ev.loc[j_plug_cond, 'energy'] = 6 * ev[j_plug_cond]['duration_hours']
        ev.loc[n_plug_cond, 'energy'] = 1.5 * ev[n_plug_cond]['duration_hours']
        pre_load_na = len(ev)
        ev = ev[~(ev['energy'].isna())] # drop any remaining missing values
        logger.debug(f"Dropped {pre_load_na-len(ev)} rows from EV dataset for having missing energy load after imputation")

        # Outlier detection for energy loads (log transformed) of individual charging events
        log_mad = mad_outlier_bounds(ev_train, ["log_energy"], threshold=self.cfg.mad_thresh)
        mad = { # convert back to actual energy
            "energy": {
                "min_val": np.expm1(log_mad["log_energy"]["min_val"]),
                "max_val": np.expm1(log_mad["log_energy"]["max_val"]),
            }
        }
        ev = cap_outliers_mad(ev, mad)

        ev.rename(columns={'starttime':'timestamp'}, inplace=True)
        ev.set_index('timestamp', inplace=True)
        
        # Aggregate EV data with resampling to given period
        ev_agg = (
            ev
                .resample(self.cfg.sampling_interval)["energy"]
                .sum()
                .reindex(self.full_date_range)
                .fillna(0)
                .sort_index()
        )

        ev_agg.index = pd.to_datetime(ev_agg.index)

        ev_agg_train = ev_agg.loc[self.train_date_range]

        ### Outlier detection for energy load per hour 
        # First we calculate paramaters over the train set and apply them
        # Then we apply these parameters over the entire set to detect outliers for the test set.
        logger.debug(f"Beginning outlier detection for train set energy load using rolling MAD method")
        ev_agg_train, mad_params = rolling_mad_outliers(ev_agg_train, k=self.cfg.mad_thresh, min_samples=self.cfg.min_outlier_samples)
        logger.debug(f"Beginning outlier detection for test set energy load using rolling MAD method - reusing parameters calculated for train set")
        ev_agg_test,_ = rolling_mad_outliers(ev_agg, k=self.cfg.mad_thresh, min_samples=0, precomputed_params=mad_params)
        ev_agg_test = ev_agg_test.loc[self.test_date_range]

        ev_agg_out = pd.concat([ev_agg_train,ev_agg_test])


        ev_agg_out.rename(columns={'outlier':'energy_outlier'}, inplace=True)

        self.ev = ev_agg_out
        self.ev.to_csv(self.cfg.ev_proc_path, index_label='timestamp')


    def _process_weather(self):        

        # Import weather data
        weather = pd.read_csv(self.cfg.weather_int_path, parse_dates=['starttime','endtime'], low_memory=False)
        
        # Duration of weather event
        weather['duration'] = (weather['endtime'] - weather['starttime']) // pd.Timedelta(minutes=1)

        # Function to drop duplicates/missings/anomalous rows etc, enrich and then split into train/test
        weather, weather_train = clean_enrich_split(
            df=weather,
            df_name='weather',
            keep_cols=self.cfg.weather_cols,
            positive_cols=['duration'],
            log_cols=['duration'],
            type_filt=self.cfg.weather_type_filt,
            split_date=self.cfg.split_date
    )

        # Outlier detection for weather event durations (log transformed) - bounds are defined using train dataset
        log_mad = mad_outlier_bounds(weather_train, ["log_duration"], threshold=self.cfg.mad_thresh)
        mad = { # convert back to actual duration
            "duration": {
                "min_val": np.expm1(log_mad["log_duration"]["min_val"]),
                "max_val": np.expm1(log_mad["log_duration"]["max_val"]),
            }
        }
        weather = cap_outliers_mad(weather, mad) # apply across whole set

        # Explode each weather event such that it is split out over the hour(s) it occurs during, with duration allocated appropriately
        w_expl = pd.DataFrame(columns=['timestamp','type','severity','duration'])
        for index, row in weather[['starttime','type','severity','duration']].iterrows():
            # Get set of exploded rows
            expl_rows = split_event_by_hour(row)
            # Add to full df of weather events
            w_expl = pd.concat([w_expl, expl_rows], axis=0)

        # Aggregate weather event data
        weather_agg = (
            w_expl.pivot_table(index=w_expl['timestamp'], columns=['type','severity'], values='duration', aggfunc='sum', fill_value=0)
                .reindex(self.full_date_range)
                .fillna(0)
                .sort_index()
        )

        weather_agg.index = pd.to_datetime(weather_agg.index)

        weather_agg.columns = ['_'.join(map(str, col)).lower() for col in weather_agg.columns] # Deconstruct column index
        weather_agg.columns = ['w_'+col+'_dur' for col in weather_agg.columns] # Indicate duration in columns names and add prefix to denote a weather column

        # Aggregate weather event duration by weather event type
        weather_agg = aggregate_features(df=weather_agg, out_name='w_rain_dur', substr1='rain')
        weather_agg = aggregate_features(df=weather_agg, out_name='w_fog_dur', substr1='fog')

        self.weather = weather_agg

        self.weather.to_csv(self.cfg.weather_proc_path, index_label='timestamp')


    def _process_temperature(self):

        ### 1. Clean Temperature Data ###

        temperature = pd.read_csv(self.cfg.temp_path, parse_dates=['starttime'], low_memory=False)

        # Function to drop duplicates/missings/anomalous rows etc, enrich and then split into train/test
        temp, temp_train = clean_enrich_split(
            df=temperature,
            df_name='temperature',
            split_date=self.cfg.split_date
        )

        # Remove values of temperature outside reasonable range
        temp_bound = (-1 <= temp['temp']) & (temp['temp'] <= 50)
        temp.loc[~temp_bound, 'temp'] = np.nan

        temp.rename(columns={'starttime':'timestamp'}, inplace=True)
        temp.set_index('timestamp', inplace=True)

        # Aggregate temperature data
        temp_agg = (
                temp
                    .resample(self.cfg.sampling_interval)["temp"]
                    .mean()
                    .reindex(self.full_date_range)
                    .sort_index()
            )

        temp_agg.index = pd.to_datetime(temp_agg.index)

        logger.debug(f"There are {(temp_agg.isna().astype(int).sum())} rows of the temperature dataset with missing values for temp before imputation")
        temp_agg = temp_agg.to_frame()
        temp_agg['temp_imputed'] = 0
        temp_agg.loc[temp_agg['temp'].isna(), 'temp_imputed'] = 1 # any missing values for temperature will be imputed so we flag them for context

        # Add column for average temperature values
        temp_agg = avg_temp_tracker(temp_agg)

        missing_at_count = temp_agg['avg_temp'].isna().astype(int).sum()
        # Impute missing temperature values
        temp_agg['avg_temp'].ffill(inplace=True)
        logger.debug(f"Forward filled {missing_at_count} values for avg_temp in temperature dataset")
        temp_agg['temp'].fillna(temp_agg['avg_temp'], inplace=True)
        
        logger.debug(f"There are {(temp_agg['temp'].isna().astype(int).sum())} rows of the temperature dataset with missing values for temp after imputation")

        temp_agg.drop(columns=['avg_temp','hour','dayofyear'], inplace=True)

        self.temperature  = temp_agg

        self.temperature.to_csv(self.cfg.temp_proc_path, index_label='timestamp')


    def _process_traffic(self):

         # Import traffic data
        traffic = pd.read_csv(self.cfg.traffic_int_path, parse_dates=['starttime','endtime'], low_memory=False)

        # Duration of traffic event
        traffic['duration'] = (traffic['endtime'] - traffic['starttime']) // pd.Timedelta(minutes=1)

        # Function to drop duplicates/missings/anomalous rows etc, enrich and then split into train/test
        traffic, traffic_train = clean_enrich_split(
            df=traffic,
            df_name='traffic',
            keep_cols=self.cfg.traffic_cols,
            positive_cols=['duration','distance'],
            log_cols=['duration','distance'],
            type_filt=self.cfg.traffic_type_filt,
            split_date=self.cfg.split_date
        )

        # Outlier detection for traffic event durations (log transformed) - bounds are defined using train dataset
        log_mad_dur = mad_outlier_bounds(traffic_train, ["log_duration"], threshold=self.cfg.mad_thresh)
        mad_dur = { # convert back to actual duration
            "duration": {
                "min_val": np.expm1(log_mad_dur["log_duration"]["min_val"]),
                "max_val": np.expm1(log_mad_dur["log_duration"]["max_val"]),
            }
        }
        traffic = cap_outliers_mad(traffic, mad_dur) # apply across whole set

        # Outlier detection for traffic event distances (log transformed) - bounds are defined using train dataset
        log_mad_dis = mad_outlier_bounds(traffic_train, ["log_distance"], threshold=self.cfg.mad_thresh)
        mad_dis = { # convert back to actual duration
            "distance": {
                "min_val": np.expm1(log_mad_dis["log_distance"]["min_val"]),
                "max_val": np.expm1(log_mad_dis["log_distance"]["max_val"]),
            }
        }
        traffic = cap_outliers_mad(traffic, mad_dis) # apply across whole set

        # Explode each traffic event such that it is split out over the hour(s) it occurs during, with duration allocated appropriately
        t_expl = pd.DataFrame(columns=['timestamp','type','severity','duration','distance'])
        for index, row in traffic[['starttime','type','severity','distance','duration']].iterrows():
            # Get set of exploded rows
            expl_rows = split_event_by_hour(row)
            # Add to full df of traffic events
            t_expl = pd.concat([t_expl, expl_rows], axis=0)

        # Aggregate traffic event data by duration
        traffic_agg = (
            t_expl.pivot_table(index=t_expl['timestamp'], columns=['type','severity'], values=['duration','distance'], aggfunc='sum', fill_value=0)
                .reindex(self.full_date_range)
                .fillna(0)
                .sort_index()
        )

        traffic_agg.index = pd.to_datetime(traffic_agg.index)

        traffic_agg.columns = ['_'.join(map(str, col)).lower() for col in traffic_agg.columns] # Deconstruct column index
        traffic_agg.columns = ['t_'+col for col in traffic_agg.columns] # Add prefix to indicate traffic column

        # Aggregate traffic event duration and distance by traffic event type 
        traffic_agg = aggregate_features(df=traffic_agg, out_name='t_cong_dis', substr1='distance_congestion')
        traffic_agg = aggregate_features(df=traffic_agg, out_name='t_flow_dis', substr1='distance_flow')
        traffic_agg = aggregate_features(df=traffic_agg, out_name='t_cong_dur', substr1='duration_congestion')
        traffic_agg = aggregate_features(df=traffic_agg, out_name='t_flow_dur', substr1='duration_flow')

        self.traffic = traffic_agg

        self.traffic.to_csv(self.cfg.traffic_proc_path, index_label='timestamp')



    def _combine(self):

        ## 1. Validate data ##

        periods = len(self.full_date_range)
        
        # Validate EV data
        if self.cfg.run_ev:
            ev = self.ev
        else:
            ev = pd.read_csv(self.cfg.ev_proc_path, index_col='timestamp', parse_dates=['timestamp'])
        validate_time_series(ev,periods,'EV')
      
        # Validate weather data
        if self.cfg.run_weather:
            weather = self.weather
        else:
            weather = pd.read_csv(self.cfg.weather_proc_path, index_col='timestamp', parse_dates=['timestamp'])
        validate_time_series(weather,periods,'Weather')

        # Validate temperature data
        if self.cfg.run_temperature:
            temperature  = self.temperature
        else:
            temperature = pd.read_csv(self.cfg.temp_proc_path, index_col='timestamp', parse_dates=['timestamp'])
        validate_time_series(temperature,periods,'Temperature')

        # Validate traffic data
        if self.cfg.run_traffic:
            traffic  = self.traffic
        else:
            traffic = pd.read_csv(self.cfg.traffic_proc_path, index_col='timestamp', parse_dates=['timestamp'])
        validate_time_series(traffic,periods,'Traffic')

        # Ensure timestamp indexes are correct
        if not (ev.index.min() == weather.index.min() == temperature.index.min() == traffic.index.min()):
            raise IndexError(
            f"Some of the train data have misaligned start points for their date range"
        )
        if not (ev.index.max() == weather.index.max() == temperature.index.max() == traffic.index.max()):
            raise IndexError(
            f"Some of the train data have misaligned end points for their date range"
        )
        
        ## 2. Combine and save combined data ##
        combined = (
            ev
            .merge(weather, left_index=True, right_index=True, how="inner")
            .merge(temperature, left_index=True, right_index=True, how="inner")
            .merge(traffic, left_index=True, right_index=True, how="inner")
        ) 
        validate_time_series(combined,periods,'Combined')
        combined.to_csv(self.cfg.combined_path, index_label='timestamp')

        logger.info(f"Successfully validated and saved full combined processed dataset to {self.cfg.processed_data_path / 'combined_processed.csv'}")


    def run(self):
        if self.cfg.run_ev:
            logger.info("Beginning _process_ev")
            self._process_ev()
            logger.info("Finished _process_ev")
        if self.cfg.run_weather:
            logger.info("Beginning _process_weather")
            self._process_weather()
            logger.info("Finished _process_weather")
        if self.cfg.run_temperature:
            logger.info("Beginning _process_temperature")
            self._process_temperature()
            logger.info("Finished _process_temperature")
        if self.cfg.run_traffic:
            logger.info("Beginning _process_traffic")
            self._process_traffic()
            logger.info("Finished _process_traffic")
        if self.cfg.run_combine:
            logger.info("Beginning _combine")
            self._combine()
            logger.info("Finished _combine")

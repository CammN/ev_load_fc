import pandas as pd
import numpy as np
from dataclasses import dataclass 
from ev_load_fc.data.preprocessing import (
    clean_enrich_split,
    mad_outlier_bounds, 
    cap_outliers_mad, 
    avg_temp_tracker, 
    mstl_resid_outlier,
    split_event_by_hour,
    validate_time_series
)
import logging
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingPipelineConfig:
    # Paths
    processed_data_path: str
    ev_int_path: str
    weather_int_path: str
    temp_path: str
    traffic_int_path: str
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
    agg_period: str
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
             end=self.cfg.split_date-pd.to_timedelta(self.cfg.agg_period),
             freq=self.cfg.agg_period
        )  
        self.test_date_range  = pd.date_range(
            start=self.cfg.split_date,
            end=self.cfg.max_timestamp-pd.to_timedelta(self.cfg.agg_period),
            freq=self.cfg.agg_period
        ) 

    def _process_ev(self):

        ### 1. Clean Data ###

        ev = pd.read_csv(self.cfg.ev_int_path, parse_dates=['starttime','endtime','transaction_date'], low_memory=False)

        # Duration of charging events in hours
        ev['duration_hours'] = pd.to_timedelta(ev['charging_time']).dt.total_seconds() / 3600

        # Function to drop duplicates/missings/anomalous rows etc, enrich and then split into train/test
        e_train, e_test = clean_enrich_split(
            df=ev,
            df_name='EV',
            keep_cols=self.cfg.ev_cols,
            positive_cols=['duration_hours'],
            log_cols=['energy'],
            split_date=self.cfg.split_date,
            split_cross = True
        )

        ev_outputs = {}

        for split_name, ev_split in {"train": e_train,"test": e_test}.items():

            logger.info(f"Processing {split_name} split for EV data")

            ev_df = ev_split.copy()
 
            # Aproximate power of charger plugs
            ev_df['charger_kw'] = ev_df['energy'] / ev_df['duration_hours'] 
            pre_plug_na = len(ev)
            ev_df = ev_df[~(ev_df['plug_type'].isna())]
            logger.debug(f"Dropped {pre_plug_na-len(ev_df)} rows from EV {split_name} dataset for missing plug type")

            # Remove extreme values based on calculated charger plug power by the x% upper and lower quantiles 
            for pt in ev_df['plug_type'].unique():
                low_quant = ev_df[ev_df['plug_type']==pt]['charger_kw'].quantile(self.cfg.kw_quant)
                upp_quant = ev_df[ev_df['plug_type']==pt]['charger_kw'].quantile(1-self.cfg.kw_quant)
                ext_val_mask = ((low_quant >= ev_df['charger_kw']) | (ev_df['charger_kw'] >= upp_quant))
                ev_df.loc[(ev_df['plug_type']==pt) & ext_val_mask, 'kw_drop'] = 1
            pre_kw_filt = len(ev_df)
            ev_df = ev_df[ev_df['kw_drop']!=1]
            logger.debug(f"Dropped {pre_kw_filt-len(ev_df)} rows from EV {split_name} dataset for having anomalous charging power")

            # Impute missing values for energy using approximate power for charger plug type 
            j_plug_cond = (ev_df['energy'].isna()) & (ev_df['plug_type']=='J1772')
            n_plug_cond = (ev_df['energy'].isna()) & (ev_df['plug_type']=='NEMA 5-20R')
            ev_df.loc[j_plug_cond, 'energy'] = 7 * ev_df[j_plug_cond]['duration_hours']
            ev_df.loc[n_plug_cond, 'energy'] = 3 * ev_df[n_plug_cond]['duration_hours']
            pre_load_na = len(ev_df)
            ev_df = ev_df[~(ev_df['energy'].isna())] # drop any remaining missing values
            logger.debug(f"Dropped {pre_load_na-len(ev_df)} rows from EV {split_name} dataset for having missing energy load after imputation")

            # Outlier detection for energy loads (log transformed) of individual charging events
            log_mad = mad_outlier_bounds(ev_df, ["log_energy"], threshold=self.cfg.mad_thresh)
            mad = { # convert back to actual energy
                "energy": {
                    "min_val": np.expm1(log_mad["log_energy"]["min_val"]),
                    "max_val": np.expm1(log_mad["log_energy"]["max_val"]),
                }
            }
            ev_df = cap_outliers_mad(ev_df, mad)

            if split_name == 'train':
                split_date_range = self.train_date_range
            elif split_name == 'test':
                split_date_range = self.test_date_range

            ev_df.rename(columns={'starttime':'timestamp'}, inplace=True)
            ev_df.set_index(ev_df['timestamp'], inplace=True)
            
            # Aggregate EV data with resampling to given period
            ev_agg = (
                ev_df
                    .resample(self.cfg.agg_period)["energy"]
                    .sum()
                    .reindex(split_date_range)
                    .fillna(0)
                    .sort_index()
            )

            ev_agg.index = pd.to_datetime(ev_agg.index)

            # Outlier detection for energy load per hour 
            logger.debug(f"Beginning outlier detection for {split_name} set energy load using MSTL residuals")
            ev_agg_out = mstl_resid_outlier(ev_agg, k=3.5, df_name=split_name)

            ev_agg_out.rename(columns={'outlier':'energy_outlier'}, inplace=True)

            ev_outputs[split_name] = ev_agg_out.copy()

        self.ev_train = ev_outputs['train']
        self.ev_test  = ev_outputs['test']

        self.ev_train.to_csv(self.cfg.processed_data_path/'ev_train.csv', index_label='timestamp')
        self.ev_test.to_csv(self.cfg.processed_data_path/'ev_test.csv', index_label='timestamp')


    def _process_weather(self):        

        # Import weather data
        weather = pd.read_csv(self.cfg.weather_int_path, parse_dates=['starttime','endtime'], low_memory=False)
        
        # Duration of weather event
        weather['duration'] = (weather['endtime'] - weather['starttime']) // pd.Timedelta(minutes=1)

        # Function to drop duplicates/missings/anomalous rows etc, enrich and then split into train/test
        w_train, w_test = clean_enrich_split(
            df=weather,
            df_name='weather',
            keep_cols=self.cfg.weather_cols,
            positive_cols=['duration'],
            log_cols=['duration'],
            type_filt=self.cfg.weather_type_filt,
            split_date=self.cfg.split_date,
            split_cross = True
        )

        weather_outputs = {}
        # Loop over train and test sets
        for split_name, weather_split in {"train": w_train,"test": w_test}.items():

            logger.info(f"Processing {split_name} split for weather data")

            w_df = weather_split.copy()

            # Outlier detection for weather event durations (log transformed)
            log_mad = mad_outlier_bounds(w_df, ["log_duration"], threshold=self.cfg.mad_thresh)
            mad = { # convert back to actual duration
                "duration": {
                    "min_val": np.expm1(log_mad["log_duration"]["min_val"]),
                    "max_val": np.expm1(log_mad["log_duration"]["max_val"]),
                }
            }
            w_df = cap_outliers_mad(w_df, mad)

            # Explode each weather event such that it is split out over the hour(s) it occurs during, with duration allocated appropriately
            wdf_expl = pd.DataFrame(columns=['timestamp','type','severity','duration'])
            for index, row in w_df[['starttime','type','severity','duration']].iterrows():
                # Get set of exploded rows
                expl_rows = split_event_by_hour(row)
                # Add to full df of weather events
                wdf_expl = pd.concat([wdf_expl, expl_rows], axis=0)

            if split_name == 'train':
                split_date_range = self.train_date_range
            elif split_name == 'test':
                split_date_range = self.test_date_range

            # Aggregate weather event data
            wdf_events_agg = (
                wdf_expl.pivot_table(index=wdf_expl['timestamp'], columns=['type','severity'], values='duration', aggfunc='sum', fill_value=0)
                    .reindex(split_date_range)
                    .fillna(0)
                    .sort_index()
            )

            wdf_events_agg.index = pd.to_datetime(wdf_events_agg.index)

            wdf_events_agg.columns = ['_'.join(map(str, col)).lower() for col in wdf_events_agg.columns] # Deconstruct column index
            wdf_events_agg.columns = [col+'_dur' for col in wdf_events_agg.columns] # Indicate duration in columns names

            weather_outputs[split_name] = wdf_events_agg.copy()

        self.weather_train = weather_outputs['train']
        self.weather_test  = weather_outputs['test']

        self.weather_train.to_csv(self.cfg.processed_data_path/'weather_train.csv', index_label='timestamp')
        self.weather_test.to_csv(self.cfg.processed_data_path/'weather_test.csv', index_label='timestamp')


    def _process_temperature(self):

        ### 1. Clean Temperature Data ###

        temperature = pd.read_csv(self.cfg.temp_path, parse_dates=['starttime'], low_memory=False)

        # Function to drop duplicates/missings/anomalous rows etc, enrich and then split into train/test
        tp_train, tp_test = clean_enrich_split(
            df=temperature,
            df_name='temperatue',
            # keep_cols=self.cfg.temp_cols,
            split_date=self.cfg.split_date,
            split_cross = False
        )

        temperature_outputs = {}
        # Loop over train and test sets
        for split_name, temperature_split in {"train": tp_train,"test": tp_test}.items():

            logger.info(f"Processing {split_name} split for weather data")

            tp_df = temperature_split.copy()

            # Remove values of temperature outside reasonable range
            temp_bound = (-1 <= tp_df['temp']) & (tp_df['temp'] <= 50)
            tp_df.loc[~temp_bound, 'temp'] = np.nan

            if split_name == 'train':
                split_date_range = self.train_date_range
            elif split_name == 'test':
                split_date_range = self.test_date_range

            tp_df.rename(columns={'starttime':'timestamp'}, inplace=True)
            tp_df.set_index(tp_df['timestamp'], inplace=True)

            # Aggregate temperature data
            tp_df_agg = (
                    tp_df
                        .resample(self.cfg.agg_period)["temp"]
                        .mean()
                        .reindex(split_date_range)
                        .sort_index()
                )

            tp_df_agg.index = pd.to_datetime(tp_df_agg.index)

            logger.debug(f"There are {(tp_df_agg.isna().astype(int).sum())} rows of the temperature {split_name} dataset with missing values for temp before imputation")

            # Add column for average temperature values
            tp_df_agg = avg_temp_tracker(tp_df_agg)

            missing_at_count = tp_df_agg['avg_temp'].isna().astype(int).sum()
            # Impute missing temperature values
            tp_df_agg['avg_temp'].ffill(inplace=True)
            logger.debug(f"Forward filled {missing_at_count} values for avg_tmp in temperature {split_name} datase")
            tp_df_agg['temp'].fillna(tp_df_agg['avg_temp'], inplace=True)
            
            logger.debug(f"There are {(tp_df_agg['temp'].isna().astype(int).sum())} rows of the temperature {split_name} dataset with missing values for temp after imputation")

            tp_df_agg.drop(columns=['avg_temp','hour','dayofyear'], inplace=True)

            temperature_outputs[split_name] = tp_df_agg.copy()
        
        self.temp_train = temperature_outputs['train']
        self.temp_test  = temperature_outputs['test']

        self.temp_train.to_csv(self.cfg.processed_data_path/'temperature_train.csv', index_label='timestamp')
        self.temp_test.to_csv(self.cfg.processed_data_path/'temperature_test.csv', index_label='timestamp')


    def _process_traffic(self):

         # Import traffic data
        traffic = pd.read_csv(self.cfg.traffic_int_path, parse_dates=['starttime','endtime'], low_memory=False)

        # Duration of traffic event
        traffic['duration'] = (traffic['endtime'] - traffic['starttime']) // pd.Timedelta(minutes=1)

        # Function to drop duplicates/missings/anomalous rows etc, enrich and then split into train/test
        tf_train, tf_test = clean_enrich_split(
            df=traffic,
            df_name='traffic',
            keep_cols=self.cfg.traffic_cols,
            positive_cols=['duration','distance'],
            log_cols=['duration','distance'],
            type_filt=self.cfg.traffic_type_filt,
            split_date=self.cfg.split_date,
            split_cross = True
        )

        traffic_outputs = {}
        # Loop over train and test sets
        for split_name, traffic_split in {"train": tf_train,"test": tf_test}.items():

            logger.info(f"Processing {split_name} split for traffic data")

            tdf = traffic_split.copy()

            # Outlier detection for traffic event durations (log transformed)
            log_mad_dur = mad_outlier_bounds(tdf, ["log_duration"], threshold=self.cfg.mad_thresh)
            mad_dur = { # convert back to actual duration
                "duration": {
                    "min_val": np.expm1(log_mad_dur["log_duration"]["min_val"]),
                    "max_val": np.expm1(log_mad_dur["log_duration"]["max_val"]),
                }
            }
            tdf = cap_outliers_mad(tdf, mad_dur)

            # Outlier detection for traffic event distances (log transformed)
            log_mad_dis = mad_outlier_bounds(tdf, ["log_distance"], threshold=self.cfg.mad_thresh)
            mad_dis = { # convert back to actual duration
                "distance": {
                    "min_val": np.expm1(log_mad_dis["log_distance"]["min_val"]),
                    "max_val": np.expm1(log_mad_dis["log_distance"]["max_val"]),
                }
            }
            tdf = cap_outliers_mad(tdf, mad_dis)

            # Explode each traffic event such that it is split out over the hour(s) it occurs during, with duration allocated appropriately
            tdf_expl = pd.DataFrame(columns=['timestamp','type','severity','duration','distance'])
            for index, row in tdf[['starttime','type','severity','distance','duration']].iterrows():
                # Get set of exploded rows
                expl_rows = split_event_by_hour(row)
                # Add to full df of traffic events
                tdf_expl = pd.concat([tdf_expl, expl_rows], axis=0)

            if split_name == 'train':
                split_date_range = self.train_date_range
            elif split_name == 'test':
                split_date_range = self.test_date_range

            # Aggregate traffic event data by duration
            tp_df_agg = (
                tdf_expl.pivot_table(index=tdf_expl['timestamp'], columns=['type','severity'], values=['duration','distance'], aggfunc='sum', fill_value=0)
                    .reindex(split_date_range)
                    .fillna(0)
                    .sort_index()
            )

            tp_df_agg.index = pd.to_datetime(tp_df_agg.index)

            tp_df_agg.columns = ['_'.join(map(str, col)).lower() for col in tp_df_agg.columns] # Deconstruct column index

            traffic_outputs[split_name] = tp_df_agg.copy()

        self.traffic_train = traffic_outputs['train']
        self.traffic_test = traffic_outputs['test']

        self.traffic_train.to_csv(self.cfg.processed_data_path/'traffic_train.csv', index_label='timestamp')
        self.traffic_test.to_csv(self.cfg.processed_data_path/'traffic_test.csv', index_label='timestamp')



    def _combine(self):

        ## 1. Validate data ##
        
        # Validate EV train and test
        if self.cfg.run_ev:
            ev_train = self.ev_train
            ev_test  = self.ev_test
        else:
            ev_train = pd.read_csv(self.cfg.processed_data_path / 'ev_train.csv', index_col='timestamp', parse_dates=['timestamp'])
            ev_test  = pd.read_csv(self.cfg.processed_data_path / 'ev_test.csv', index_col='timestamp', parse_dates=['timestamp'])
        validate_time_series(ev_train,'train','EV', train_range=self.train_date_range, test_range=self.test_date_range)
        validate_time_series(ev_test,'test','EV', train_range=self.train_date_range, test_range=self.test_date_range)

        # Validate weather train and test
        if self.cfg.run_weather:
            weather_train = self.weather_train
            weather_test  = self.weather_test
        else:
            weather_train = pd.read_csv(self.cfg.processed_data_path / 'weather_train.csv', index_col='timestamp', parse_dates=['timestamp'])
            weather_test = pd.read_csv(self.cfg.processed_data_path / 'weather_test.csv', index_col='timestamp', parse_dates=['timestamp'])
        validate_time_series(weather_train,'train','Weather', train_range=self.train_date_range, test_range=self.test_date_range)
        validate_time_series(weather_test,'test','Weather', train_range=self.train_date_range, test_range=self.test_date_range)

        # Validate temperature train and test
        if self.cfg.run_temperature:
            temp_train = self.temp_train
            temp_test  = self.temp_test
        else:
            temp_train = pd.read_csv(self.cfg.processed_data_path / 'temperature_train.csv', index_col='timestamp', parse_dates=['timestamp'])
            temp_test = pd.read_csv(self.cfg.processed_data_path / 'temperature_test.csv', index_col='timestamp', parse_dates=['timestamp'])
        validate_time_series(temp_train,'train','Temperature', train_range=self.train_date_range, test_range=self.test_date_range)
        validate_time_series(temp_test,'test','Temperature', train_range=self.train_date_range, test_range=self.test_date_range)

        # Validate traffic train and test
        if self.cfg.run_traffic:
            traffic_train = self.traffic_train
            traffic_test  = self.traffic_test
        else:
            traffic_train = pd.read_csv(self.cfg.processed_data_path / 'traffic_train.csv', index_col='timestamp', parse_dates=['timestamp'])
            traffic_test = pd.read_csv(self.cfg.processed_data_path / 'traffic_test.csv', index_col='timestamp', parse_dates=['timestamp'])
        validate_time_series(traffic_train,'train','Traffic', train_range=self.train_date_range, test_range=self.test_date_range)
        validate_time_series(traffic_test,'test','Traffic', train_range=self.train_date_range, test_range=self.test_date_range)

        # Ensure timestamp indexes are correct
        if not (ev_train.index.min() == weather_train.index.min() == temp_train.index.min() == traffic_train.index.min()):
            raise IndexError(
            f"Some of the train data have misaligned start points for their date range"
        )
        if not (ev_train.index.min() == weather_train.index.min() == temp_train.index.min() == traffic_train.index.min()):
            raise IndexError(
            f"Some of the train data have misaligned end points for their date range"
        )
        if not (ev_test.index.min() == weather_test.index.min() == temp_test.index.min() == traffic_test.index.min()):
            raise IndexError(
            f"Some of the test data have misaligned start points for their date range"
        )
        if not (ev_test.index.min() == weather_test.index.min() == temp_test.index.min() == traffic_test.index.min()):
            raise IndexError(
            f"Some of the test data have misaligned end points for their date range"
        )
        
        ## 2. Combine and save train data ##
        train = (
            ev_train
            .merge(weather_train, left_index=True, right_index=True, how="inner")
            .merge(temp_train, left_index=True, right_index=True, how="inner")
            .merge(traffic_train, left_index=True, right_index=True, how="inner")
        ) 
        validate_time_series(train,'train','Combined', train_range=self.train_date_range, test_range=self.test_date_range)
        train.to_csv(self.cfg.processed_data_path / 'train.csv', index_label='timestamp')

        logger.info(f"Successfully validated and saved train dataset to {self.cfg.processed_data_path / 'train.csv'}")

        ## 3. Combine and save test data ##
        test = (
            ev_test
            .merge(weather_test, left_index=True, right_index=True, how="inner")
            .merge(temp_test, left_index=True, right_index=True, how="inner")
            .merge(traffic_test, left_index=True, right_index=True, how="inner")
        ) 
        validate_time_series(test,'test','Combined', self.train_date_range, self.test_date_range)

        test.to_csv(self.cfg.processed_data_path / 'test.csv', index_label='timestamp')

        logger.info(f"Successfully validated and saved test dataset to {self.cfg.processed_data_path / 'test.csv'}")
        


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

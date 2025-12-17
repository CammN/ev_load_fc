import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calender
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import MSTL
from scipy.stats import skew
from ev_load_fc.config import CFG, resolve_path
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Paths
interim_data_path   = resolve_path(CFG["paths"]["interim_data"])
processed_data_path = resolve_path(CFG["paths"]["processed_data"])
ev_int_path         = interim_data_path / CFG["files"]["ev_filt_filename"]
weather_int_path    = interim_data_path / CFG["files"]["weather_filt_filename"]
temp_path           =  interim_data_path / CFG['files']['temperature_filename']
traffic_int_path    = interim_data_path / CFG["files"]["traffic_filt_filename"]
# Filters
min_timestamp       = pd.to_datetime(CFG["data"]['raw_filters']["min_timestamp"])
max_timestamp       = pd.to_datetime(CFG["data"]['raw_filters']["max_timestamp"])
# Preprocessing parameters
kw_quant            = CFG["data"]['preprocessing']["plug_power_quantile_bound"]
mad_thresh          = CFG["data"]['preprocessing']["mad_thresholds"]
split_date          = pd.to_datetime(CFG["data"]['preprocessing']["split_date"])
agg_period          = CFG["data"]['preprocessing']["aggregation_period"]
train_date_range    = pd.date_range(start=min_timestamp, end=split_date-pd.to_timedelta(agg_period), freq=agg_period)
test_date_range     = pd.date_range(start=split_date, end=max_timestamp-pd.to_timedelta(agg_period), freq=agg_period)
holidays            = list(CFG["data"]['preprocessing']["holidays"])


def mad_outlier_bounds(df, cols:list=[], threshold:int=3)->dict:
    """Calculate Mean Absolute Deviation (MAD) outlier bounds for specified column(s) in a DataFrame or Series.

    Args:
        df (pd.DataFrame, pd.Series): Input DataFrame or Series.
        cols (list, optional): List of columns to calculate MAD bounds for.
        threshold (int, optional): Threshold multiplier for MAD to define outlier bounds.

    Returns:
        dict: Dictionary containing MAD bounds for each specified column.
    """
    if len(cols)>0:
        MAD_df = pd.DataFrame(df[cols]).copy()
    else:
        MAD_df = df.copy()

    # Instantiate dict for storing MAD values for each column
    MAD_dict = {}

    # Iterate over every column we are checking for outliers
    for col in MAD_df.columns:
        MAD_dict[col] = {}
        series = MAD_df[col]

        # For MAD outlier detection we require the column to contain at least 3 unique values
        if series.nunique(dropna=False)>2:
            median = series.median()
            MAD_dict[col]['median'] = median

            # For columns with non-skewed distributions use a single MAD for the upper and lower bounds
            if abs(skew(series)) < 1:
                MAD = (series-median).abs().median()
                MAD_dict[col]['skewed'] = False
                MAD_dict[col]['max_val'] = median + (threshold * MAD)
                MAD_dict[col]['min_val'] = median - (threshold * MAD)
            # For columns with skewed distributions use a separate MAD for the upper and lower bounds
            else: 
                upper_series = series[series>=median]
                lower_series = series[series<median]
                upper_MAD = (upper_series-median).abs().median()
                lower_MAD = (lower_series-median).abs().median()
                MAD_dict[col]['skewed'] = True
                MAD_dict[col]['max_val'] = median + (threshold * upper_MAD)
                MAD_dict[col]['min_val'] = median - (threshold * lower_MAD)
            
        else:
            MAD_dict[col]['skewed'] = False
            MAD_dict[col]['max_val'] = np.inf
            MAD_dict[col]['min_val'] = -np.inf
        
    return MAD_dict


def cap_outliers_mad(df:pd.DataFrame, mad_dict:dict)->pd.DataFrame:
    """Cap outliers in DataFrame based on Mean Absolute Difference (MAD) bounds.

    Args:
        df (pd.DataFrame): Input DataFrame with potential outliers.
        mad_dict (dict): Dictionary containing MAD bounds for each column.

    Returns:
        pd.DataFrame: DataFrame with outliers capped.
    """

    df_capped = df.copy()

    for col in mad_dict.keys():
        max_val = mad_dict[col]['max_val']
        min_val = mad_dict[col]['min_val']

        lower_cond = df_capped[col] < min_val
        upper_cond = df_capped[col] > max_val

        cap_count = lower_cond.astype(int).sum() + upper_cond.astype(int).sum() 
        print(f"Capping {cap_count} outlier values for {col} column")

        df_capped.loc[upper_cond, col] = max_val
        df_capped.loc[lower_cond, col] = min_val

    return df_capped


def avg_temp_tracker(series):
    """
    Adds a column for average temperature across an expanding window up to the point in time of each row, to be used for imputation.
    Averages are calculated per hour per day of the year.
    """

    df_at = pd.DataFrame(series)
    df_at['hour'] = df_at.index.hour
    df_at['dayofyear'] = df_at.index.dayofyear

    # Initialise dict containing average temperatures and counts used to calculate them
    avg_temps = {
        (hour, day): {'avg_tmp':np.nan,'count':0}
        for hour in df_at['hour'].unique()
        for day in df_at['dayofyear'].unique()
    }

    df_at['avg_temp'] = 0

    # Iterate through every row and populate avg_temp column
    for index, row in df_at.iterrows():
        hour = row['hour']
        day  = row['dayofyear']
        temp = row['temp']

        avg_temp = avg_temps[(hour,day)]['avg_tmp']
        
        if pd.isna(temp): # If temp is missing then we don't update the average temp for this hour/day
            df_at.loc[index, 'avg_temp'] = avg_temp
        else:
            avg_temps[(hour,day)]['count'] += 1
            curr_count =  avg_temps[(hour,day)]['count']

            if np.isnan(avg_temp):
                avg_temp = 0

            new_avg_tmp = (avg_temp * (curr_count-1) + temp) / curr_count

            df_at.loc[index, 'avg_temp'] = new_avg_tmp
            avg_temps[(hour,day)]['avg_tmp'] = new_avg_tmp

    return df_at


def get_holidays(holiday_subset):

    # Initialise US holidays data
    cal = calender()
    holidays = cal.holidays(start=min_timestamp, end=max_timestamp, return_name=True)
    holidays_df = holidays.reset_index().rename(columns={'index':'ds', 0:'holiday'})

    # Add extra holidays
    extra_hols = pd.DataFrame({
            'ds': [
                datetime(2017,12,24),datetime(2018,12,24),datetime(2019,12,24),datetime(2017,12,26),datetime(2018,12,26),datetime(2019,12,26)
            ],
            'holiday': [
                'Christmas Eve','Christmas Eve','Christmas Eve','Boxing Day','Boxing Day','Boxing Day'
            ]
        }   
    )
    holidays_df = pd.concat([holidays_df, extra_hols], axis=0)

    # Expand holidays df to hourly basis
    hours = pd.DataFrame({'hour': range(24)})
    hourly_holidays_df = holidays_df.merge(hours, how='cross')
    hourly_holidays_df['ds'] = hourly_holidays_df['ds'] + pd.to_timedelta(hourly_holidays_df['hour'], unit='h')

    # Create hourly index across our date range
    hourly_index = pd.date_range(start=min_timestamp, end=max_timestamp, freq='1h')  
    ts = pd.DataFrame(index=hourly_index)
    print(f"Range from {ts.index.min()} to {ts.index.max()}")

    # Create One Hot Encodings for each holiday
    holiday_ohe = pd.get_dummies(hourly_holidays_df.set_index('ds')['holiday'], dtype=int)
    holiday_hourly = holiday_ohe.reindex(ts.index, fill_value=0)
    holiday_hourly_cut = holiday_hourly[holiday_subset] # subset

    return holiday_hourly_cut

    
def mstl_resid_outlier(series:pd.Series, k:float=3.5, df_name:str='') -> pd.DataFrame:
    
    mstl_start = time.time()
    mstl_model = MSTL(series, periods=[24,24*7], stl_kwargs={'robust':True}).fit()
    resid = mstl_model.resid

    median = np.median(resid)
    mad = np.median(np.abs(resid - median))
    sigma = 1.4826 * mad
    outlier_mask = np.abs(resid - median) > k * sigma

    df = pd.DataFrame(series)
    df['outlier'] = 0
    df.loc[outlier_mask, 'outlier'] = 1

    mstl_end = time.time()

    if len(df_name)>0:
        col_string = f"from {df_name.strip()} data "
    else:
        col_string = ""
    print(f"% of {series.name} column {col_string}flagged as outliers: {(df['outlier'].sum()/len(df['outlier'])):.2f}%")
    print(f"MSTL outlier detection completed in {mstl_end-mstl_start:.2f}s")

    return df


def split_event_by_hour(row):
    
    expl_rows = []

    duration_left = row['duration']
    current_time  = row['starttime']
    other_cols = {} # Treat all other column values as static
    for col in set(row.index) - set(['duration','starttime']):
        other_cols[col] = row[col]

    while duration_left > 0:
        # End of the current hour
        hour_end = (current_time.floor("h") + pd.Timedelta(hours=1))

        # Minutes available in this hour
        minutes_in_hour = (hour_end - current_time).total_seconds() / 60

        # Allocate duration for current hour
        hour_duration = min(duration_left, minutes_in_hour)

        dyn_cols = {
            "timestamp": current_time.floor("h"),
            "duration": hour_duration
        }
        all_cols = dyn_cols | other_cols

        expl_rows.append(all_cols)

        # Go to next hour
        duration_left -= hour_duration
        current_time = hour_end

    return pd.DataFrame(expl_rows)



############################################
###      Meta preprocessing functions    ###
############################################

def ev_clean_split():

    ### 1. Clean Data ###

    ev = pd.read_csv(ev_int_path, parse_dates=['start_date','end_date','transaction_date'])

    # Drop duplicate rows
    predup = len(ev)
    ev.drop_duplicates(inplace=True)
    print(f"Dropped {predup-len(ev)} duplicate rows from EV dataset")

    # Key columns
    ev_keep_cols = ['start_date','energy','charging_time', 'plug_type']
    ev = ev[ev_keep_cols]

    # Drop rows with missing start dates
    missing_sd_mask = ev['start_date'].isna()
    ev = ev[~missing_sd_mask]
    print(f"Dropped {missing_sd_mask.astype(int).sum()} rows from EV dataset for having missing values for start_date")

    # Set start_date as index
    ev.rename(columns={'start_date':'timestamp'}, inplace=True)
    if not pd.api.types.is_datetime64_any_dtype(ev["timestamp"]):
        ev['timestamp'] = pd.to_datetime(ev['timestamp'])
    ev.set_index(ev['timestamp'], inplace=True)
    # Ensure chronology
    ev.sort_index(inplace=True)
    
    # Duration of charging events in hours
    ev['duration_hours'] = pd.to_timedelta(ev['charging_time']).dt.total_seconds() / 3600
    pre_dur_filt = len(ev)
    ev = ev[ev['duration_hours'] > 0] 
    print(f"Dropped {pre_dur_filt-len(ev)} rows from EV dataset for having non-positive durations of charging events")
    
    # Aproximate power of charger plugs
    ev['charger_kw'] = ev['energy'] / ev['duration_hours'] 
    pre_plug_na = len(ev)
    ev = ev[~(ev['plug_type'].isna())]
    print(f"Dropped {pre_plug_na-len(ev)} rows from EV dataset for missing plug type")

    # Remove extreme values based on calculated charger plug power by the x% upper and lower quantiles 
    for pt in ev['plug_type'].unique():
        low_quant = ev[ev['plug_type']==pt]['charger_kw'].quantile(kw_quant)
        upp_quant = ev[ev['plug_type']==pt]['charger_kw'].quantile(1-kw_quant)
        ext_val_mask = ((low_quant >= ev['charger_kw']) | (ev['charger_kw'] >= upp_quant))
        ev.loc[(ev['plug_type']==pt) & ext_val_mask, 'kw_drop'] = 1
    pre_kw_filt = len(ev)
    ev = ev[ev['kw_drop']!=1]
    print(f"Dropped {pre_kw_filt-len(ev)} rows from EV dataset for having anomalous charging power")
    
    # Remove anomalous values with non-positive energy load
    pre_load_filt = len(ev)
    ev = ev[ev['energy'] > 0]
    print(f"Dropped {pre_load_filt-len(ev)} rows from EV dataset for having non-positive energy load")

    # Impute missing values for energy using approximate power for charger plug type 
    j_plug_cond = (ev['energy'].isna()) & (ev['plug_type']=='J1772')
    n_plug_cond = (ev['energy'].isna()) & (ev['plug_type']=='NEMA 5-20R')
    ev.loc[j_plug_cond, 'energy'] = 7 * ev[j_plug_cond]['duration_hours']
    ev.loc[n_plug_cond, 'energy'] = 3 * ev[n_plug_cond]['duration_hours']
    pre_load_na = len(ev)
    ev = ev[~(ev['energy'].isna())] # drop any remaining missing values
    print(f"Dropped {pre_load_na-len(ev)} rows from EV dataset for having missing energy load after imputation")

    ev['log_energy'] = np.log1p(ev['energy'])

    ### 2. Split data and process ###

    ev_train = ev[ev.index <  split_date]
    ev_test  = ev[ev.index >= split_date]

    outputs = {}

    for split_name, ev_split in {"train": ev_train,"test": ev_test}.items():

        print(f"Processing {split_name} split for EV data")

        evdf = ev_split.copy()

        # Outlier detection for energy loads (log transformed) of individual charging events
        log_mad = mad_outlier_bounds(evdf, ["log_energy"], threshold=mad_thresh)

        mad = { # convert back to actual energy
            "energy": {
                "min_val": np.expm1(log_mad["log_energy"]["min_val"]),
                "max_val": np.expm1(log_mad["log_energy"]["max_val"]),
            }
        }

        evdf = cap_outliers_mad(evdf, mad)

        if split_name == 'train':
            split_date_range = train_date_range
        elif split_name == 'test':
            split_date_range = test_date_range
        
        # Aggregate EV data with resampling to given period
        ev_agg = (
            evdf
                .resample(agg_period)["energy"]
                .sum()
                .reindex(split_date_range)
                .fillna(0)
                .sort_index()
        )

        ev_agg.index = pd.to_datetime(ev_agg.index)

        # Outlier detection for energy load per hour 
        print(f"Beginning outlier detection for {split_name} set energy load using MSTL residuals")
        ev_agg_out = mstl_resid_outlier(ev_agg, k=3.5, df_name=split_name)

        ev_agg_out.rename(columns={'outlier':'energy_outlier'}, inplace=True)

        outputs[split_name] = ev_agg_out.copy()
    
    ev_train_agg = outputs['train']
    ev_test_agg  = outputs['test']

    ev_train_agg.to_csv(processed_data_path/'ev_train.csv', index_label='timestamp')
    ev_test_agg.to_csv(processed_data_path/'ev_test.csv', index_label='timestamp')
    

def weather_clean_split():

    ### 1. Clean Weather Event Data ###

    # Import weather data
    weather = pd.read_csv(weather_int_path, parse_dates=['starttime','endtime'])

    # Drop duplicate rows
    predup = len(weather)
    weather.drop_duplicates(inplace=True)
    print(f"Dropped {predup-len(weather)} duplicate rows from weather dataset")

    # Key columns
    weather_keep_cols = ['starttime','endtime','type','severity']
    weather = weather[weather_keep_cols]

    # Drop rows with missing start times and end times
    missing_st_mask = weather['starttime'].isna()
    weather = weather[~missing_st_mask]
    print(f"Dropped {missing_st_mask.astype(int).sum()} rows from weather dataset for having missing values for starttime")
    missing_et_mask = weather['endtime'].isna()
    weather = weather[~missing_et_mask]
    print(f"Dropped {missing_et_mask.astype(int).sum()} rows from weather dataset for having missing values for endtime")
    
    # Duration of weather event
    weather['duration'] = (weather['endtime'] - weather['starttime']) // pd.Timedelta(minutes=1)

    # Remove "Cold" weather event type - anomalous given the climate of PA
    wtype_mask = weather['type'] == 'Cold'
    weather = weather[~wtype_mask]
    print(f"Dropped {wtype_mask.astype(int).sum()} rows from weather dataset for having a 'Cold' event type")

    # Remove anomalous values with non-positive durations
    pre_dur_filt = len(weather)
    weather = weather[weather['duration'] > 0]
    print(f"Dropped {pre_dur_filt-len(weather)} rows from weather dataset for having non-positive event durations")

    weather['log_dur'] = np.log1p(weather['duration'])


    ### 2. Split and Process Weather Events Data ###

    weather_train = weather[(weather['endtime'] < split_date) | ((weather['starttime'] <= split_date) & (weather['endtime'] > split_date))]
    weather_test  = weather[weather['starttime'] >= split_date]

    weather_outputs = {}

    # Loop over train and test sets
    for split_name, weather_split in {"train": weather_train,"test": weather_test}.items():

        print(f"Processing {split_name} split for weather data")

        wdf = weather_split.copy()

        # Outlier detection for weather event durations (log transformed)
        log_mad = mad_outlier_bounds(wdf, ["log_dur"], threshold=mad_thresh)
        mad = { # convert back to actual duration
            "duration": {
                "min_val": np.expm1(log_mad["log_dur"]["min_val"]),
                "max_val": np.expm1(log_mad["log_dur"]["max_val"]),
            }
        }
        wdf = cap_outliers_mad(wdf, mad)

        # Explode each weather event such that it is split out over the hour(s) it occurs during, with duration allocated appropriately
        wdf_expl = pd.DataFrame(columns=['timestamp','type','severity','duration'])
        for index, row in wdf[['starttime','type','severity','duration']].iterrows():
            # Get set of exploded rows
            expl_rows = split_event_by_hour(row)
            # Add to full df of weather events
            wdf_expl = pd.concat([wdf_expl, expl_rows], axis=0)

        if split_name == 'train':
            split_date_range = train_date_range
        elif split_name == 'test':
            split_date_range = test_date_range

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

    weather_train_agg = weather_outputs['train']
    weather_test_agg  = weather_outputs['test']
    
    weather_train_agg.to_csv(processed_data_path/'weather_train.csv', index_label='timestamp')
    weather_test_agg.to_csv(processed_data_path/'weather_test.csv', index_label='timestamp')


def temperature_clean_split():

    ### 1. Clean Temperature Data ###

    temperature = pd.read_csv(temp_path, parse_dates=['starttime'])

    # Drop duplicate rows
    predup = len(temperature)
    temperature.drop_duplicates(inplace=True)
    print(f"Dropped {predup-len(temperature)} duplicate rows from temperature dataset")

    # Drop rows with missing start times and end times
    missing_st_mask = temperature['starttime'].isna()
    temperature = temperature[~missing_st_mask]
    print(f"Dropped {missing_st_mask.astype(int).sum()} rows from temperatures dataset for having missing values for starttime")

    temperature.rename(columns={'starttime':'timestamp'}, inplace=True)
    temperature.set_index(temperature['timestamp'], inplace=True)

    # Remove values of temperature outside reasonable range
    temp_bound = (-1 <= temperature['temp']) & (temperature['temp'] <= 50)
    temperature.loc[~temp_bound, 'temp'] = np.nan

    ### 3. Split and Process Temperature Data ###

    temperature_train = temperature[temperature.index < split_date]
    temperature_test  = temperature[temperature.index >= split_date]

    temp_outputs = {}

    # Loop over train and test sets
    for split_name, temp_split in {"train": temperature_train,"test": temperature_test}.items():

        print(f"Processing {split_name} split for temperature data")

        if split_name == 'train':
            split_date_range = train_date_range
        elif split_name == 'test':
            split_date_range = test_date_range

        # Aggregate temperature data
        tdf_agg = (
                temp_split
                    .resample(agg_period)["temp"]
                    .mean()
                    .reindex(split_date_range)
                    .sort_index()
            )

        tdf_agg.index = pd.to_datetime(tdf_agg.index)

        print(f"There are {(tdf_agg.isna().astype(int).sum())} rows of the temperature {split_name} dataset with missing values for temp before imputation")

        # Add column for average temperature values
        tdf_agg = avg_temp_tracker(tdf_agg)

        missing_at_count = tdf_agg['avg_temp'].isna().astype(int).sum()
        # Impute missing temperature values
        tdf_agg['avg_temp'].ffill(inplace=True)
        print(f"Forward filled {missing_at_count} values for avg_tmp in temperature {split_name} datase")
        tdf_agg['temp'].fillna(tdf_agg['avg_temp'], inplace=True)
        
        print(f"There are {(tdf_agg['temp'].isna().astype(int).sum())} rows of the temperature {split_name} dataset with missing values for temp after imputation")

        tdf_agg.drop(columns=['avg_temp','hour','dayofyear'], inplace=True)

        temp_outputs[split_name] = tdf_agg.copy()
    
    temp_train_agg = temp_outputs['train']
    temp_test_agg  = temp_outputs['test']

    temp_train_agg.to_csv(processed_data_path/'temperature_train.csv', index_label='timestamp')
    temp_test_agg.to_csv(processed_data_path/'temperature_test.csv', index_label='timestamp')


def traffic_clean_split():

    # 1. Clean Data #

    # Import traffic data
    traffic = pd.read_csv(traffic_int_path, parse_dates=['starttime','endtime'])

    # Drop duplicate rows
    predup = len(traffic)
    traffic.drop_duplicates(inplace=True)
    print(f"Dropped {predup-len(traffic)} duplicate rows from traffic dataset")

    # Key columns
    traffic_keep_cols = ['starttime','endtime','type','severity','distance']
    traffic = traffic[traffic_keep_cols]

    # Drop rows with missing start times and end times
    missing_st_mask = traffic['starttime'].isna()
    traffic = traffic[~missing_st_mask]
    print(f"Dropped {missing_st_mask.astype(int).sum()} rows from traffic dataset for having missing values for starttime")
    missing_et_mask = traffic['endtime'].isna()
    traffic = traffic[~missing_et_mask]
    print(f"Dropped {missing_et_mask.astype(int).sum()} rows from traffic dataset for having missing values for endtime")

    # # Set starttime as timestamp index
    # traffic.rename(columns={'starttime':'timestamp'})
    # if not pd.api.types.is_datetime64_any_dtype(traffic["timestamp"]):
    #     traffic['timestamp'] = pd.to_datetime(traffic['timestamp'])
    # traffic.set_index(traffic['timestamp'])

    # Duration of traffic event
    traffic['duration'] = (traffic['endtime'] - traffic['starttime']) // pd.Timedelta(minutes=1)

    # Keep certain types of traffic event
    ttype_mask = ~(traffic['type'].isin(['Congestion','Flow-Incident','Event']))
    traffic = traffic[~ttype_mask]
    print(f"Dropped {(ttype_mask).astype(int).sum()} rows from traffic dataset for not being 'Congestion', 'Flow-Incident' or 'Event' event type")
    
    traffic = traffic[traffic['type'].isin(['Congestion','Flow-Incident','Event'])]

    # Remove anomalous values with non-positive durations
    pre_dur_filt = len(traffic)
    traffic = traffic[traffic['duration'] > 0]
    print(f"Dropped {pre_dur_filt-len(traffic)} rows from traffic dataset for having non-positive event durations")
    # Remove anomalous values with negative distances
    pre_dis_filt = len(traffic)
    traffic = traffic[traffic['distance'] >= 0]
    print(f"Dropped {pre_dis_filt-len(traffic)} rows from traffic dataset for having negative event distances")

    traffic['log_dur'] = np.log1p(traffic['duration'])
    traffic['log_dis'] = np.log1p(traffic['distance'])

    ### 2. Split and Process traffic Events Data ###

    traffic_train = traffic[(traffic['endtime'] < split_date) | ((traffic['starttime'] <= split_date) & (traffic['endtime'] > split_date))]
    traffic_test  = traffic[traffic['starttime'] >= split_date]

    traffic_outputs = {}

    # Loop over train and test sets
    for split_name, traffic_split in {"train": traffic_train,"test": traffic_test}.items():

        print(f"Processing {split_name} split for traffic data")

        tdf = traffic_split.copy()

        # Outlier detection for traffic event durations (log transformed)
        log_mad_dur = mad_outlier_bounds(tdf, ["log_dur"], threshold=mad_thresh)
        mad_dur = { # convert back to actual duration
            "duration": {
                "min_val": np.expm1(log_mad_dur["log_dur"]["min_val"]),
                "max_val": np.expm1(log_mad_dur["log_dur"]["max_val"]),
            }
        }
        tdf = cap_outliers_mad(tdf, mad_dur)

        # Outlier detection for traffic event distances (log transformed)
        log_mad_dis = mad_outlier_bounds(tdf, ["log_dis"], threshold=mad_thresh)
        mad_dis = { # convert back to actual duration
            "distance": {
                "min_val": np.expm1(log_mad_dis["log_dis"]["min_val"]),
                "max_val": np.expm1(log_mad_dis["log_dis"]["max_val"]),
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
            split_date_range = train_date_range
        elif split_name == 'test':
            split_date_range = test_date_range

        # Aggregate traffic event data by duration
        tdf_agg = (
            tdf_expl.pivot_table(index=tdf_expl['timestamp'], columns=['type','severity'], values=['duration','distance'], aggfunc='sum', fill_value=0)
                .reindex(split_date_range)
                .fillna(0)
                .sort_index()
        )

        tdf_agg.index = pd.to_datetime(tdf_agg.index)

        tdf_agg.columns = ['_'.join(map(str, col)).lower() for col in tdf_agg.columns] # Deconstruct column index

        traffic_outputs[split_name] = tdf_agg.copy()

    traffic_train_agg = traffic_outputs['train']
    traffic_test_agg  = traffic_outputs['test']
    
    traffic_train_agg.to_csv(processed_data_path/'traffic_train.csv', index_label='timestamp')
    traffic_test_agg.to_csv(processed_data_path/'traffic_test.csv', index_label='timestamp')

    return traffic_train_agg, traffic_test_agg


def validate_time_series(df, dset, name):

    if dset == 'train':
        periods = len(train_date_range)
    elif dset =='test':
        periods = len(test_date_range)
    
    if len(df) != periods:
        raise ValueError( 
        f"{name} {dset} dataset has incorrect number of total periods, expected {periods} but got {len(df)}"
    )
    if df.index.nunique() != periods:
        raise ValueError( 
        f"{name} {dset} dataset has incorrect number of unique periods, expected {periods} but got {df.index.nunique()}"
    )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
        f"{name} {dset} dataset index must be DatetimeIndex, got {type(df.index)}"
    )

    print(f"{name} {dset} dataset is valid across the given date range")


def combine_to_model_set():

    ## 1. Validate train data ##
    
    ev_train = pd.read_csv(processed_data_path / 'ev_train.csv', index_col='timestamp', parse_dates=['timestamp'])
    validate_time_series(ev_train,'train','EV')

    weather_train = pd.read_csv(processed_data_path / 'weather_train.csv', index_col='timestamp', parse_dates=['timestamp'])
    validate_time_series(weather_train,'train','Weather')

    temperature_train = pd.read_csv(processed_data_path / 'temperature_train.csv', index_col='timestamp', parse_dates=['timestamp'])
    validate_time_series(temperature_train,'train','Temperature')

    traffic_train = pd.read_csv(processed_data_path / 'traffic_train.csv', index_col='timestamp', parse_dates=['timestamp'])
    validate_time_series(traffic_train,'train','Traffic')

    if not (ev_train.index.min() == weather_train.index.min() == temperature_train.index.min() == traffic_train.index.min()):
        raise IndexError(
        f"Some of the train data have misaligned start points for their date range"
    )
    if not (ev_train.index.min() == weather_train.index.min() == temperature_train.index.min() == traffic_train.index.min()):
        raise IndexError(
        f"Some of the train data have misaligned end points for their date range"
    )
    
    ## 2. Combine train data ##

    train = (
        ev_train
        .merge(weather_train, left_index=True, right_index=True, how="inner")
        .merge(temperature_train, left_index=True, right_index=True, how="inner")
        .merge(traffic_train, left_index=True, right_index=True, how="inner")
    ) 
    validate_time_series(train,'train','Combined')

    train.to_csv(processed_data_path / 'train.csv', index_label='timestamp')

    ## 3. Validate test data ##
    
    ev_test = pd.read_csv(processed_data_path / 'ev_test.csv', index_col='timestamp', parse_dates=['timestamp'])
    validate_time_series(ev_test,'test','EV')

    weather_test = pd.read_csv(processed_data_path / 'weather_test.csv', index_col='timestamp', parse_dates=['timestamp'])
    validate_time_series(weather_test,'test','Weather')

    temperature_test = pd.read_csv(processed_data_path / 'temperature_test.csv', index_col='timestamp', parse_dates=['timestamp'])
    validate_time_series(temperature_test,'test','Temperature')

    traffic_test = pd.read_csv(processed_data_path / 'traffic_test.csv', index_col='timestamp', parse_dates=['timestamp'])
    validate_time_series(traffic_test,'test','Traffic')

    if not (ev_test.index.min() == weather_test.index.min() == temperature_test.index.min() == traffic_test.index.min()):
        raise IndexError(
        f"Some of the test data have misaligned start points for their date range"
    )
    if not (ev_test.index.min() == weather_test.index.min() == temperature_test.index.min() == traffic_test.index.min()):
        raise IndexError(
        f"Some of the train data have misaligned end points for their date range"
    )
    
    ## 4. Combine test data ##

    test = (
        ev_test
        .merge(weather_test, left_index=True, right_index=True, how="inner")
        .merge(temperature_test, left_index=True, right_index=True, how="inner")
        .merge(traffic_test, left_index=True, right_index=True, how="inner")
    ) 
    validate_time_series(test,'test','Combined')

    test.to_csv(processed_data_path / 'test.csv', index_label='timestamp')
    
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import MSTL
from scipy.stats import skew
from typing import Tuple
from collections import defaultdict
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logger = logging.getLogger(__name__)


def clean_enrich_split(
        df:pd.DataFrame, 
        df_name:str,
        keep_cols:list = [], 
        positive_cols:list = [],
        log_cols:list = [],
        type_filt:list = [],
        split_date:datetime = [], 
):
    """
    Clean, enrich, and split a DataFrame to retrieve train set.

    Args:
        df (pd.DataFrame): Input DataFrame to be processed
        df_name (str): Name of the DataFrame (for logging purposes)
        keep_cols (list, optional): Columns to keep in the DataFrame.
        positive_cols (list, optional): Columns that should be positive values.
        log_cols (list, optional): Columns to apply log transformation.
        type_filt (list, optional): Event types to filter.
        split_date (datetime, optional): Date to split the data.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): First element is the processed data across the entire time period, second element is the processed data across the train set's time period
    """

    logger.debug(f"Beginning cleaning of {df_name} with {len(df)} rows")

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Drop fully missing rows
    df = df[~(df.isna().all(axis=1))]

    # Restrict to columns of interest
    if len(keep_cols) > 0:
        df = df[keep_cols]

    # Drop rows with missing start times (and end dates if exists)
    missing_sd_mask = df['starttime'].isna()
    df = df[~missing_sd_mask]
    logger.debug(f"Dropped {missing_sd_mask.astype(int).sum()} rows from {df_name} dataset for having missing values for starttime")
    if 'endtime' in df.columns:
        missing_et_mask = df['endtime'].isna()
        df = df[~missing_et_mask]
        logger.debug(f"Dropped {missing_et_mask.astype(int).sum()} rows from {df_name} dataset for having missing values for endtime")

    # Remove non-positive anomalous values for given columns
    if len(positive_cols) > 0:
        for col in positive_cols:
            df = df[~(df[col].isna())]
            pre_anom_filt = len(df)
            df = df[df[col] > 0]
            logger.debug(f"Dropped {pre_anom_filt-len(df)} rows from {df_name} dataset for having non-positive {col}")

    # Restrict to given event types
    if len(type_filt) > 0:
        df = df[df['type'].isin(type_filt)]

    # Apply log transformation to specified columns
    if len(log_cols) > 0:
        for col in log_cols:
            df = df[~(df[col].isna())]
            df[f"log_{col}"] = np.log1p(df[col])


    if 'endtime' in df.columns:
        df_train = df[df['endtime'] < split_date]
    else:
        df_train = df[df['starttime'] < split_date]

    return df, df_train


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
        logger.debug(f"Capping {cap_count} outlier values for {col} column")

        df_capped.loc[upper_cond, col] = max_val
        df_capped.loc[lower_cond, col] = min_val

    return df_capped


def avg_temp_tracker(series) -> pd.DataFrame:
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


def rolling_mad_outliers(
        series:pd.Series, 
        k:float=3.5, 
        min_samples:int=0, 
        precomputed_params:dict=None
    ) -> Tuple[pd.DataFrame, dict]:
    """
    Detects outliers in a time series using a rolling Mean Absolute Deviation (MAD) approach.
    Outliers are defined as points that lie beyond k times the estimated standard deviation from the median.
    The median and standard deviation estimates are computed separately for each (hour, weekday) pair to account for seasonality.

    Args:
        series (pd.Series): Time series to compute outliers for.
        k (float, optional): _description_. Default: 3.5.
        min_samples (int, optional): Minimum number of samples in the time series to start checking for outliers. Defaults to 0.
            - If precomputed_params is provided this is ignored and outlier detection starts from the beginning of the series.
        precomputed_params (dict, optional): Contains precomputed values for the median ad sigma for each hour-weekday pair. Defaults to None.
            - If this is not provided the function will compute and update these values as it iterates through the time series.

    Raises:
        IndexError: Input time series must have a DatetimeIndex.

    Returns:
        Tuple[pd.DataFrame, dict]: _description_
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise IndexError("Time series must have a Pandas DatetimeIndex")

    # Convert series to df and extract hour/weekday
    target = series.name
    df = pd.DataFrame(series)
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek

    hour_weekday_values = defaultdict(list)
    
    # If not provided, initialise dict containing a rolling history of all values for each (hour,weekday) pair as well as their median and standard deviation estimates
    hour_weekday_params = (
        precomputed_params
        if precomputed_params is not None
        else defaultdict(lambda: {'median': np.nan, 'sigma': np.nan})
    )

    if precomputed_params is not None:
        min_samples = 0

    ### Iterate over every row, applying estimates for median and standard deviation (sigma) and using them to detect outliers
    ### Update said estimates if a dict of their values was not already provided
    df_out = df.copy()
    df_out['outlier'] = 0
    for sample, (dtindex, row) in enumerate(df.iterrows()):
        # Initialise values for row
        key      = (row['hour'],row['weekday'])
        val      = row[target]
        all_vals = hour_weekday_values[key]
        median   = hour_weekday_params[key]['median']
        sigma    = hour_weekday_params[key]['sigma']
        
        # Check if current value lies outside bounds defined above, if so => outlier
        # Also ensure we only define outliers from a minimum number of samples onwards (early detection will be unstable)
        if sample >= min_samples:
            if (np.abs(val - median) > k * sigma) and (sigma > 0):
                df_out.at[dtindex, 'outlier'] = 1

        # After detecting for outliers we can update our value set
        all_vals.append(val) 

        # If a parameter dict was not passed we update the estimates for our median and sigma
        if precomputed_params is None:
            # Compute components of robust Z-score
            median = np.median(all_vals)
            mad    = np.median(np.abs(np.array(all_vals) - median))
            sigma  = 1.4826 * mad if mad > 0 else np.nan # estimation for the standard deviation

            # Update dict containing historical parameters
            hour_weekday_params[key]['median'] = median
            hour_weekday_params[key]['sigma']  = sigma

    logger.debug(f"% of {target} column flagged as outliers: {(df_out['outlier'].sum()/len(df_out['outlier'])):.2f}%")

    return df_out, dict(hour_weekday_params)



def mstl_resid_outliers(series:pd.Series, k:float=3.5, df_name:str='', params:dict=None) -> pd.DataFrame:
    """
    Detects outliers in a time series by applying Season-Trend decomposition with LOESS for multiple seasonalities.
    Outliers are defined to exist upper and lower bounds for the residuals of the decomposition.
    Bounds on the residuals are calculated using an estimate of their standard deviation based off their Mean Absolute Deviation. 

    Args:
        series (pd.Series): Time series.
        k (float, optional): Threshold multiplier for SD estimation. Default: 3.5
        df_name (str, optional): Name of time series for logging purposes. Default: ''
        params (dict, optional): Contains pre-computed values for the median and estimated standard deviation. Default: None

    Returns:
        pd.DataFrame: _description_
    """

    # Perform MSTL and extract reiduals
    mstl_start = time.time()
    mstl_model = MSTL(series, periods=[24,24*7], stl_kwargs={'robust':True}).fit()
    resid = mstl_model.resid

    # Assign/compute median and sigma
    if params:
        median = params['median']
        sigma  = params['sigma']
    else:
        median = np.median(resid)
        mad = np.median(np.abs(resid - median))
        sigma = 1.4826 * mad # estimation for the standard deviation

    # Define outlier range and flag
    outlier_mask = np.abs(resid - median) > k * sigma
    df = pd.DataFrame(series)
    df['outlier'] = 0
    df.loc[outlier_mask, 'outlier'] = 1

    mstl_end = time.time()

    if len(df_name)>0:
        col_string = f"from {df_name.strip()} data "
    else:
        col_string = ""
    logger.debug(f"% of {series.name} column {col_string}flagged as outliers: {(df['outlier'].sum()/len(df['outlier'])):.2f}%")
    logger.debug(f"MSTL outlier detection completed in {mstl_end-mstl_start:.2f}s")

    return df, sigma, median


def split_event_by_hour(row:pd.Series) -> pd.DataFrame:
    """
    Row-wise callback function to explode event rows across hourly intervals.

    Args:
        row (pd.Series): Contains event information.

    Returns:
        pd.DataFrame: Exploded event row.
    """

    expl_rows = []

    duration_left = row['duration'] # Total length of event
    current_time  = row['starttime'] # Start time of event
    other_cols = {} # Treat all other column values as static
    for col in set(row.index) - set(['duration','starttime']):
        other_cols[col] = row[col]

    # Create new rows until we have covered the entire duration period
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


def validate_time_series(df:pd.DataFrame, periods:int, name:str):
    """
    Validates that a time series DataFrame has the correct number of total and unique periods, and that its index is a DatetimeIndex.

    Args:
        df (pd.DataFrame): Input DataFrame to validate.
        periods (int): Expected number of periods.
        name (str): Name of the DataFrame (for logging purposes).

    Raises:
        ValueError: If df has the incorrect number of total periods
        ValueError: If df has the incorrect number of unqiue periods
        TypeError: If df does not have an index of type DatetimeIndex
    """
    if len(df) != periods:
        raise ValueError( 
        f"{name} dataset has incorrect number of total periods, expected {periods} but got {len(df)}"
    )
    if df.index.nunique() != periods:
        raise ValueError( 
        f"{name} dataset has incorrect number of unique periods, expected {periods} but got {df.index.nunique()}"
    )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
        f"{name} dataset index must be DatetimeIndex, got {type(df.index)}"
    )

    logger.debug(f"{name} dataset is valid across the given date range")


def scale_features(X):
    return X
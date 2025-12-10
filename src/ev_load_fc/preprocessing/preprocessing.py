import pandas as pd
from scipy.stats import skew
from ev_load_fc.config import CFG, resolve_path
# Paths
interim_data_path   =  resolve_path(CFG["paths"]["interim_data"])
processed_data_path =  resolve_path(CFG["paths"]["processed_data"])
ev_int_path         =  interim_data_path / CFG["files"]["ev_filt_filename"]
weather_int_path    =  interim_data_path / CFG["files"]["weather_filt_filename"]
traffic_int_path    =  interim_data_path / CFG["files"]["traffic_filt_filename"]
ev_pro_path         =  processed_data_path / CFG["files"]["ev_filename"]
weather_pro_path    =  processed_data_path / CFG["files"]["weather_filename"]
traffic_pro_path    =  processed_data_path / CFG["files"]["traffic_filename"]
# Preprocessing parameters
min_kw     =   CFG["preprocessing"]['cleaning']["min_charger_kw"]
max_kw     =   CFG["preprocessing"]['cleaning']["max_charger_kw"] 
mad_thresh =   CFG["preprocessing"]['cleaning']["mad_thresholds"]



def mad_outlier_bounds(df, cols:list=[], threshold=3)->dict:
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
            
        # else:
        #     MAD_dict[col]['skewed'] = False
        #     MAD_dict[col]['max_val'] = np.inf
        #     MAD_dict[col]['min_val'] = -np.inf
        
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

        df_capped.loc[df_capped[col] > max_val, col] = max_val
        df_capped.loc[df_capped[col] < min_val, col] = min_val

    return df_capped


### Meta preprocessing functions ###

def ev_clean_enrich_split():

    ev_filt = pd.read_csv(ev_int_path, parse_dates=['start_date','end_date','transaction_date'])

    ev_keep_cols = ['start_date','energy','charging_time', 'plug_type']
    ev_filt = ev_int_path[ev_keep_cols]

    # Drop rows with missing start dates
    missing_sd_mask = ev_filt['start_date'].isna()
    ev_filt = ev_filt[missing_sd_mask]
    print(f"Dropped {missing_sd_mask.astype(int).sum()} rows from EV dataset for having missing values for start_date.")

    # Set start_date as index
    ev_filt.rename(['start_date'])
    ev_filt.set_index(ev_filt['start_date'])

    # Remove extreme values based on calculated charger plug power
    ev_filt['duration_hours'] = pd.to_timedelta(ev_filt['charging_time']).dt.total_seconds() / 3600
    ev_filt['charger_kw'] = ev_filt['energy'] / ev_filt['duration_hours']
    ext_val_mask = ev_filt[(min_kw < ev_filt['charger_kw']) & (ev_filt['charger_kw'] < max_kw)]
    ev_filt[ext_val_mask]

    # Impute missing values for energy using approximate power for charger plug type 
    ev_filt.loc[(ev_filt['energy'].isna()) & (ev_filt['plug_type']=='J1772'), 'energy'] = 7 * ['duration_hours']
    ev_filt.loc[(ev_filt['energy'].isna()) & (ev_filt['plug_type']=='NEMA 5-20R'), 'energy'] = 3 * ['duration_hours']
    ev_filt = ev_filt[ev_filt['energy'].isna()] # drop any remaining missing values

    # Remove anomalous values based on negative energy load
    neg_eng_mask = ev_filt[ev_filt['charger_kw'] < 0]
    ev_filt[neg_eng_mask]

    # Finding and capping outliers for energy load using MAD thresholds
    ev_mad_dict = mad_outlier_bounds(ev_filt, ['energy'], mad_thresh)
    ev_filt = cap_outliers_mad(ev_filt, ev_mad_dict)



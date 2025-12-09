import pandas as pd
from scipy.stats import skew


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
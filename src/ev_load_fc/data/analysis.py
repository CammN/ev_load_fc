import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller

def ts_completeness(df:pd.DataFrame, period:str, n:int, time_col:str)->float:
    """Calculate the completeness ratio of periods for given time series.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        period (str): Time frequency period (e.g., "h" for hourly, "D" for daily).
        n (int): Number of periods to consider.
        time_col (str): Name of the time column in the DataFrame.

    Returns:
        float: Completeness ratio of the time series data.
    """

    # Group by the specified time frequency and count occurrences
    df_np  = (df.groupby(pd.Grouper(key=time_col, freq=f"{n}{period}"))
              [time_col]
              .count()
              )
    # Calculate the total number of expected periods
    np_range = df_np.index.max()-df_np.index.min()
    np_periods = np_range / np.timedelta64(n, period)
    # Calculate the number of unique datetime periods with non-zero counts
    unique_datetimes = df_np[df_np.values>0].count()
    # Ratio of unique datetimes periods to expected periods
    completeness_ratio = unique_datetimes / np_periods

    return completeness_ratio


def plot_time_series(df:pd.DataFrame, period_col:str, agg_col:str, weekday_split:bool=False, weekday_list:list=[0,1,2,3,4,5,6], agg_type="total")->None:
    """Plot a time series of a columns aggregated over a specified period.

    Args:
        df (pd.DataFrame): Time series data with datetime index and energy values.
        period_col (str): Period column to group by (e.g. "hourly_datetime", "date", "month").
        agg_col (str): Column of df to aggregate
        weekday_split (bool, optional): Whether to split the plot by weekdays.
        weekday_list (int, optional): Specific weekdays to plot if weekday_split is True.
        agg_type (str, optional): "total" to sum aggregation column, "mean" to take mean of aggregation column. 

    Returns:
        None: Displays the plot.
    """

    weekdays = {0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"}

    ts = df.groupby(by=[period_col.lower()],sort=True)[agg_col].sum()

    plt.figure(figsize=(10, 5))
    if weekday_split:
        for i in range(7):
            if i in weekday_list:
                weekday_condition = df["weekday"] == i
                if agg_type.lower() == "total":
                    ts = df[weekday_condition].groupby(by=[period_col.lower()],sort=True)[agg_col].sum()
                elif agg_type.lower() == "mean":
                    ts = df[weekday_condition].groupby(by=[period_col.lower()],sort=True)[agg_col].mean()
                if isinstance(ts.index, pd.PeriodIndex):
                    ts.index = ts.index.to_timestamp()
                plt.plot(ts.index, ts.values, label=weekdays[i])
    else:
        if agg_type == "total":
            ts = df.groupby(by=[period_col.lower()],sort=True)[agg_col].sum()
        elif agg_type == "mean":
            ts = df.groupby(by=[period_col.lower()],sort=True)[agg_col].mean()
        if isinstance(ts.index, pd.PeriodIndex):
                    ts.index = ts.index.to_timestamp()
        plt.plot(ts.index, ts.values)
    plt.title(f"{agg_type.upper()} {agg_col.upper()} Over Time")
    plt.xlabel("Date")
    plt.ylabel(agg_col)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend() 
    plt.tight_layout()
    plt.show()


def check_adf_stationarity(series:pd.Series, regression_type:str="c"):
    """
    Returns set of Augmented- Dick-Fuller statistics to assess the stationarity of a time series.
    Both constant and constant+trend stationarity can be assessed

    Args:
        series (pd.Series): Time series to assess stationarity
        regression_type (str, optional): Constant/trend order to include in regression. Default: "c".
                                            - "c" = constant only
                                            - "ct" = constant and trend
    """

    # Reference: https://machinelearningmastery.com/time-series-data-stationary-python/

    result = adfuller(series.values, regression=regression_type)

    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))

    if (result[1] <= 0.05) & (result[4]["5%"] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")


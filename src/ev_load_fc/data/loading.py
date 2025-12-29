# Packages
import pandas as pd
from datetime import datetime
from meteostat import Hourly 
import pandas as pd
import logging
logger = logging.getLogger(__name__)


def col_standardisation(df:pd.DataFrame)->pd.DataFrame:
    """
    Standardise column names by applying consistent formatting.
    Removes suffixes contained in parentheses.
    Removes blanks.
    Enforces lower case for all characters.
    Replaces non-standard characters with an underscore.

    Args:
        df (pd.DataFrame): DataFrame with raw column names.

    Returns:
        pd.DataFrame: DataFrame with standardised column names.
    """
    df_renamed = df.copy()

    for col_name in df_renamed.columns:
        if '(' in col_name:
            last_index = col_name.find('(')
        else:
            last_index = len(col_name)
            
        new_col_name = (col_name[:last_index]
                        .strip()
                        .lower()
                        .replace(' ', '_')
                        .replace('-', '_')
                        .replace('/', '_') 
                        .replace(':', '_') 
                        )
        df_renamed = df_renamed.rename(columns={col_name: new_col_name})

    return df_renamed


def filtered_chunking(csv_path:str, 
                      start_date_col:str, 
                      end_date_col:str, 
                      date_format:str, 
                      chunksize:int=10000, 
                      min_date=datetime(1900,1,1,0,0,0), 
                      max_date=datetime(2100,1,1,0,0,0), 
                      state_list:list=[], 
                      county_list:list=[], 
                      city_list:list=[]) -> pd.DataFrame:
    """ 
    Read in large CSV file using chunking, filtering chunks by date range and location.

    Args:
        csv_path (str): Path to the CSV file.
        start_date_col (str): Name of the start date column.
        end_date_col (str): Name of the end date column.
        date_format (str): strftime of date columns to filter by.
        chunksize (int, optional): Number of rows per chunk. Default: 10000
        min_date (datetime, optional): Minimum date for filtering. Default: 1st January 1900
        max_date (datetime, optional): Maximum date for filtering. Default: 1st January 2100
        state_list (list, optional): List of states to filter by.
        county_list (list, optional): List of counties to filter by.
        city_list (list, optional): List of cities to filter by.


    Returns:
        pd.DataFrame: Filtered DataFrame.
    """

    # Initialise chunking
    chunks = pd.read_csv(csv_path, chunksize=chunksize)
    filtered_chunks = []

    # Iterate over all chunks and filter based on conditions
    for c_num, chunk in enumerate(chunks):
        chunk_condition = (pd.to_datetime(chunk[start_date_col],format=date_format, errors='coerce') >= min_date) \
                            & (pd.to_datetime(chunk[end_date_col],format=date_format, errors='coerce') < max_date)
        
        # Extend filtering conditions based on state, county and city lists
        if len(state_list)>0:
            chunk_condition = chunk_condition & (chunk['State'].isin(state_list))
        if len(county_list)>0:
            chunk_condition = chunk_condition & (chunk['County'].isin(county_list))
        if len(city_list)>0:
            chunk_condition = chunk_condition & (chunk['City'].isin(city_list))
            
        f_chunk = chunk[chunk_condition]

        filtered_chunks.append(f_chunk)

    # Concat all filtered chunks
    chunked_data = pd.concat(filtered_chunks, ignore_index=True)

    csv_str = str(csv_path)
    filename = csv_str[csv_str.rfind("\\") + 1:]
    logger.info(f"Finished chunking {filename}")
    logger.info(f"{len(chunked_data)} rows were loaded over {c_num+1} chunks") 
    
    # Standardise date columns to datetime "YYYY-mm-dd HH:MM:SS"
    # Start date
    chunked_data[start_date_col] = pd.to_datetime(chunked_data[start_date_col],format=date_format, errors='coerce')
    chunked_data[start_date_col] = chunked_data[start_date_col].dt.strftime("%Y-%m-%d %H:%M:%S")
    chunked_data[start_date_col] = pd.to_datetime(chunked_data[start_date_col]) 
    # End date
    chunked_data[end_date_col] = pd.to_datetime(chunked_data[end_date_col],format=date_format, errors='coerce')
    chunked_data[end_date_col] = chunked_data[end_date_col].dt.strftime("%Y-%m-%d %H:%M:%S")
    chunked_data[end_date_col] = pd.to_datetime(chunked_data[end_date_col]) 

    return chunked_data


def meteo_stat_temp(stations:list, min_ts:pd.Timestamp, max_ts:pd.Timestamp) -> pd.DataFrame:
    """
    Fetches weather station temperature data using the MeteoStat API.

    Args:
        stations (list): List of weather station names to fetch data from.
        min_ts (pd.Timestamp): Minimum timestamp of data to fetch.
        max_ts (pd.Timestamp): Maximum timestamp of data to fetch.

    Returns:
        pd.DataFrame: Temperature DataFrame
    """

    # Import weather data on hourly basis
    hourly_data = Hourly(stations, min_ts, max_ts)
    meteo_data = hourly_data.fetch()
    # Extract temperature data
    temp_data = meteo_data[['temp']].reset_index().rename(columns={'station':'temp_ws', 'time':'starttime'})

    return temp_data



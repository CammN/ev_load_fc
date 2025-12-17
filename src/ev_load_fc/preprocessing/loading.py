# Packages
import pandas as pd
from datetime import datetime
from meteostat import Hourly 
import pandas as pd
from ev_load_fc.config import CFG, resolve_path
# Paths
raw_data_path     =  resolve_path(CFG['paths']['raw_data'])
interim_data_path =  resolve_path(CFG['paths']['interim_data'])
ev_raw_path       =  raw_data_path / CFG['files']['ev_filename']
weather_raw_path  =  raw_data_path / CFG['files']['weather_filename']
traffic_raw_path  =  raw_data_path / CFG['files']['traffic_filename']
ev_int_path       =  interim_data_path / CFG['files']['ev_filt_filename']
weather_int_path  =  interim_data_path / CFG['files']['weather_filt_filename']
temp_path         =  interim_data_path / CFG['files']['temperature_filename']
traffic_int_path  =  interim_data_path / CFG['files']['traffic_filt_filename']
# Filters 
min_timestamp     =  pd.to_datetime(CFG['data']['raw_filters']['min_timestamp'])
max_timestamp     =  pd.to_datetime(CFG['data']['raw_filters']['max_timestamp'])
weather_cities    =  CFG['data']['raw_filters']['weather_cities']
mts_stations      =  CFG['data']['raw_filters']['meteostat_staions']
traffic_cities    =  CFG['data']['raw_filters']['traffic_cities']


def col_standardisation(df:pd.DataFrame)->pd.DataFrame:
    """Standardise column names by applying consistent formatting.

    Args:
        df (pd.DataFrame): The DataFrame whose columns are to be standardised.

    Returns:
        pd.DataFrame: The DataFrame with standardised column names.
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
    """ Filter large CSV file in chunks by date range and county list.

    Args:
        csv_path (str): Path to the CSV file.
        start_date_col (str): Name of the start date column.
        end_date_col (str): Name of the end date column.
        date_format (str): strftime
        chunksize (int, optional): Number of rows per chunk. 
        min_date (datetime, optional): Minimum date for filtering. 
        max_date (datetime, optional): Maximum date for filtering.
        state_list (list, optional): List of states to filter by.
        county_list (list, optional): List of counties to filter by.
        city_list (list, optional): List of cities to filter by.


    Returns:
        pd.DataFrame: Filtered DataFrame containing only the relevant chunks.
    """

    # Initialise chunking
    chunks = pd.read_csv(csv_path, chunksize=chunksize)
    filtered_chunks = []

    # Iterate over all chunks and filter based on conditions
    for c_num, chunk in enumerate(chunks):
        print(f'Chunk: {c_num}')
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

        print(f"Size of filtered chunk: {len(f_chunk)}")
        filtered_chunks.append(f_chunk)

    # Concat all filtered chunks
    chunked_data = pd.concat(filtered_chunks, ignore_index=True)
    
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


def meteo_stat_temp(stations, min_ts, max_ts):

    # Import weather data on hourly basis
    hourly_data = Hourly(stations, min_ts, max_ts)
    meteo_data = hourly_data.fetch()
    # Extract temperature data
    temp_data = meteo_data[['temp']].reset_index().rename(columns={'station':'temp_ws', 'time':'starttime'})

    return temp_data


### Meta loading functions ###

def filt_save_ev():
        
    # Trim EV data
    ev_data_trim = filtered_chunking(ev_raw_path, 
                                    start_date_col='Start Date', 
                                    end_date_col='End Date',
                                    date_format='%m/%d/%Y %H:%M',
                                    chunksize=100000, 
                                    min_date=min_timestamp, 
                                    max_date=max_timestamp)
    # Standardise column names
    ev_data_trim = col_standardisation(ev_data_trim)
    # Save
    ev_data_trim.to_csv(ev_int_path, index=False)


def filt_save_weather():        
    # Trim weather data
    weather_data_trim = filtered_chunking(weather_raw_path, 
                                    start_date_col='StartTime(UTC)', 
                                    end_date_col='EndTime(UTC)',
                                    date_format='%Y-%m-%d %H:%M:%S',
                                    chunksize=100000, 
                                    min_date=min_timestamp, 
                                    max_date=max_timestamp,
                                    city_list=weather_cities)
    # Standardise column names
    weather_data_trim = col_standardisation(weather_data_trim)

    # Import temperature data from meteostat
    temp_data = meteo_stat_temp(mts_stations,min_timestamp,max_timestamp)
    # Merge with LSTW weather dataset

    # Save
    weather_data_trim.to_csv(weather_int_path, index=False)
    temp_data.to_csv(temp_path, index=False)


def filt_save_traffic():     
    # Trim traffic data
    traffic_data_trim = filtered_chunking(traffic_raw_path, 
                                    start_date_col='StartTime(UTC)', 
                                    end_date_col='EndTime(UTC)',
                                    date_format='%Y-%m-%d %H:%M:%S', 
                                    chunksize=100000,
                                    min_date=min_timestamp, 
                                    max_date=max_timestamp,
                                    city_list=traffic_cities)
    # Standardise column names
    traffic_data_trim = col_standardisation(traffic_data_trim)
    # Save
    traffic_data_trim.to_csv(traffic_int_path, index=False)
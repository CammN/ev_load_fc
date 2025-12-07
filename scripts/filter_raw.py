# This script allows a user to read in and filter 1-3 of the raw datasets based on date ranges and state/county/city
# It then standardises all column names and saves as CSV files

import argparse
import pandas as pd
from ev_load_fc.config import CFG, resolve_path
from ev_load_fc.data.loading import filtered_chunking, col_standardisation

# Paths
raw_data_path = resolve_path(CFG["paths"]["raw_data"])
interim_data_path = resolve_path(CFG["paths"]["interim_data"])
ev_data_path = raw_data_path / CFG["data"]["ev_filename"]
weather_data_path = raw_data_path / CFG["data"]["weather_filename"]
traffic_data_path = raw_data_path / CFG["data"]["traffic_filename"]
# Filters
min_timestamp = pd.to_datetime(CFG["preprocessing"]['raw_filters']["min_timestamp"])
max_timestamp = pd.to_datetime(CFG["preprocessing"]['raw_filters']["max_timestamp"])
weather_cities = CFG["preprocessing"]['raw_filters']["weather_cities"]
traffic_cities = CFG["preprocessing"]['raw_filters']["traffic_cities"]


def main(ev, weather, traffic):
    
    if ev:
        # Trim EV data
        ev_data_trim = filtered_chunking(ev_data_path, 
                                        start_date_col='Start Date', 
                                        end_date_col='End Date',
                                        date_format='%m/%d/%Y %H:%M',
                                        chunksize=100000, 
                                        min_date=min_timestamp, 
                                        max_date=max_timestamp)
        # Standardise column names
        ev_data_trim = col_standardisation(ev_data_trim)
        # Save
        ev_data_trim.to_csv(interim_data_path/'ev_data_filt_date.csv', index=False)
        print("----------------------------------------------")
        print("Successfully saved trimmed EV data")

    if weather:
        # Trim weather data
        weather_data_trim = filtered_chunking(weather_data_path, 
                                        start_date_col='StartTime(UTC)', 
                                        end_date_col='EndTime(UTC)',
                                        date_format='%Y-%m-%d %H:%M:%S',
                                        chunksize=100000, 
                                        min_date=min_timestamp, 
                                        max_date=max_timestamp,
                                        city_list=weather_cities)
        # Standardise column names
        weather_data_trim = col_standardisation(weather_data_trim)
        # Save
        weather_data_trim.to_csv(interim_data_path/'weather_data_filt_date_city.csv', index=False)
        print("----------------------------------------------")
        print("Successfully saved trimmed weather data")

    if traffic:
        # Trim traffic data
        traffic_data_trim = filtered_chunking(traffic_data_path, 
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
        traffic_data_trim.to_csv(interim_data_path/'traffic_data_filt_date_city.csv', index=False)
        print("----------------------------------------------")
        print("Successfully saved trimmed traffic data")
    

def parse_args():
    """Parse command line arguments for processing raw datasets."""
    parser = argparse.ArgumentParser(description="Trim raw EV, weather, and traffic data into interim datasets.")
    parser.add_argument( "--ev", action="store_true", help="Process EV data.")
    parser.add_argument("--weather", action="store_true", help="Process weather data.")
    parser.add_argument( "--traffic", action="store_true", help="Process traffic data.")

    return parser.parse_args()


if __name__ == "__main__":
    # args if user wants to process a specific dataset
    args = parse_args()
    if not (args.ev or args.weather or args.traffic):
        args.ev = args.weather = args.traffic = True

    main(ev=args.ev, weather=args.weather, traffic=args.traffic)
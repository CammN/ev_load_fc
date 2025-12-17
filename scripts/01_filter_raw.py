# This script allows a user to read in and filter 1-3 of the raw datasets based on date ranges and state/county/city
# Additional temperature data is brought in via the meteostat public API and merged onto the weather dataset
# It then standardises all column names and saves as CSV files

import argparse
from ev_load_fc.preprocessing.loading import filt_save_ev, filt_save_weather, filt_save_traffic


def main(ev, weather, traffic):
    
    if ev:
        filt_save_ev()
        print("----------------------------------------------")
        print("Successfully saved trimmed EV data")

    if weather:
        filt_save_weather()
        print("----------------------------------------------")
        print("Successfully saved trimmed LSTW weather + meteostat temperature data")

    if traffic:
        filt_save_traffic()
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
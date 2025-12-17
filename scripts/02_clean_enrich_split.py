import argparse
from ev_load_fc.config import CFG, resolve_path
from ev_load_fc.preprocessing.preprocessing import ev_clean_split, weather_clean_split, temperature_clean_split, traffic_clean_split, combine_to_model_set
import time


def main(ev, weather, temperature, traffic, combine):

    if ev:
        print("Beginning EV data preprocessing")
        ev_start = time.time()
        ev_clean_split()
        ev_end = time.time()
        print(f"Successfully completed ev data preprocessing in {(ev_end-ev_start):.2f}s")

    if weather:
        print("Beginning weather data preprocessing")
        weather_start = time.time()
        weather_clean_split()
        weather_end = time.time()
        print(f"Successfully completed weather data preprocessing in {(weather_end-weather_start):.2f}s")

    if temperature:
        print("Beginning temperature data preprocessing")
        temp_start = time.time()
        temperature_clean_split()
        temp_end = time.time()
        print(f"Successfully completed temperature data preprocessing in {(temp_end-temp_start):.2f}s")

    if traffic:
        print("Beginning traffic data preprocessing")
        traffic_start = time.time()
        traffic_clean_split()
        traffic_end = time.time()
        print(f"Successfully completed traffic data preprocessing in {(traffic_end-traffic_start):.2f}s")

    if combine:
        print("Beginning combination process for all data")
        combine_start = time.time()
        combine_to_model_set()
        combine_end = time.time()
        print(f"Successfully completed combination in {(combine_end-combine_start):.2f}s")


def parse_args():
    """Parse command line arguments for preprocessing interim datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ev", action="store_true", help="Process EV data.")
    parser.add_argument("--weather", action="store_true", help="Process weather data.")
    parser.add_argument("--temperature", action="store_true", help="Process temperature data.")
    parser.add_argument("--traffic", action="store_true", help="Process traffic data.")
    parser.add_argument("--combine", action="store_true", help="Combine all data into final train and test sets.")

    return parser.parse_args()


if __name__ == "__main__":
    # args if user wants to process a specific dataset
    args = parse_args()
    if not (args.ev or args.weather or args.temperature or args.traffic or args.combine):
        args.ev = args.weather = args.temperature = args.traffic = args.combine = True

    # main(ev=False, weather=False, temperature=False, traffic=False, combine=True)
    main(ev=args.ev, weather=args.weather, temperature=args.temperature, traffic=args.traffic, combine=args.combine)
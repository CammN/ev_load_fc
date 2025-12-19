import pandas as pd
from dataclasses import dataclass 
from ev_load_fc.data.loading import filtered_chunking, col_standardisation, meteo_stat_temp
import logging
logger = logging.getLogger(__name__)


@dataclass
class LoadingPipelineConfig:
    # Paths
    ev_raw_path: str
    weather_raw_path: str
    traffic_raw_path: str
    ev_int_path: str
    weather_int_path: str
    traffic_int_path: str
    temp_path: str
    # Filters
    min_timestamp: pd.Timestamp
    max_timestamp: pd.Timestamp
    weather_cities: list
    traffic_cities: list
    mts_stations: list
    # Optional runtime flags
    run_ev: bool = True
    run_weather: bool = True
    run_traffic: bool = True


class LoadingPipeline:
    """Pipeline to load, filter and save EV, weather (including temperature), and traffic datasets."""


    def __init__(self, config: LoadingPipelineConfig):
        self.cfg = config


    def _process_ev(self):
        logger.info("Starting EV data processing")    
        # Trim EV data
        ev_data_trim = filtered_chunking(self.cfg.ev_raw_path, 
                                        start_date_col='Start Date', 
                                        end_date_col='End Date',
                                        date_format='%m/%d/%Y %H:%M',
                                        chunksize=100000, 
                                        min_date=self.cfg.min_timestamp, 
                                        max_date=self.cfg.max_timestamp)
        # Standardise column names
        ev_data_trim = col_standardisation(ev_data_trim)
        ev_data_trim.rename(columns={'start_date':'starttime','end_date':'endtime'}, inplace=True)
        # Save
        ev_data_trim.to_csv(self.cfg.ev_int_path, index=False)
        logger.info(f"Successfully saved trimmed EV data to {self.cfg.ev_int_path}")


    def _process_weather(self):        
        logger.info("Starting weather data processing")  
        # Trim weather data
        weather_data_trim = filtered_chunking(self.cfg.weather_raw_path, 
                                        start_date_col='StartTime(UTC)', 
                                        end_date_col='EndTime(UTC)',
                                        date_format='%Y-%m-%d %H:%M:%S',
                                        chunksize=100000, 
                                        min_date=self.cfg.min_timestamp, 
                                        max_date=self.cfg.max_timestamp,
                                        city_list=self.cfg.weather_cities)
        # Standardise column names
        weather_data_trim = col_standardisation(weather_data_trim)

        # Import temperature data from meteostat
        temp_data = meteo_stat_temp(self.cfg.mts_stations,self.cfg.min_timestamp,self.cfg.max_timestamp)

        # Save
        weather_data_trim.to_csv(self.cfg.weather_int_path, index=False)
        logger.info(f"Successfully saved trimmed weather data to {self.cfg.weather_int_path}")
        temp_data.to_csv(self.cfg.temp_path, index=False)
        logger.info(f"Successfully saved temperature data to {self.cfg.temp_path}")


    def _process_traffic(self):     
        logger.info("Starting traffic data processing")  
        # Trim traffic data
        traffic_data_trim = filtered_chunking(self.cfg.traffic_raw_path, 
                                        start_date_col='StartTime(UTC)', 
                                        end_date_col='EndTime(UTC)',
                                        date_format='%Y-%m-%d %H:%M:%S', 
                                        chunksize=100000,
                                        min_date=self.cfg.min_timestamp, 
                                        max_date=self.cfg.max_timestamp,
                                        city_list=self.cfg.traffic_cities)
        # Standardise column names
        traffic_data_trim = col_standardisation(traffic_data_trim)
        # Save
        traffic_data_trim.to_csv(self.cfg.traffic_int_path, index=False)
        logger.info(f"Successfully saved trimmed EV data to {self.cfg.traffic_int_path}")


    def run(self):

        if self.cfg.run_ev:
            self._process_ev()

        if self.cfg.run_weather:
            self._process_weather()

        if self.cfg.run_traffic:
            self._process_traffic()
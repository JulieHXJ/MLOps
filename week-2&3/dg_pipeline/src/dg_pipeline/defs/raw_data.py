
import pandas as pd
from dagster import asset

from dg_pipeline.resources.data_loader import BikeRentalDataLoader


@asset(group_name = "raw_data")
def registered_rentals(data_loader: BikeRentalDataLoader) -> pd.DataFrame:
    registered = data_loader.load_dataset("registered_rentals")
    registered["datetime"] = pd.to_datetime(registered["datetime"]) #convert datetime
    registered["hour"] = registered["datetime"].dt.floor("h") # create hour feature
    return registered

@asset(group_name = "raw_data")
def direct_pickups(data_loader: BikeRentalDataLoader) -> pd.DataFrame:
    direct = data_loader.load_dataset("direct_pickups")
    direct["datetime"] = pd.to_datetime(direct["datetime"])
    direct["hour"] = direct["datetime"].dt.floor("h")
    return direct

@asset(group_name = "raw_data")
def weather_data(data_loader: BikeRentalDataLoader) -> pd.DataFrame:
    weather = data_loader.load_dataset("weather_data")
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    weather["hour"] = weather["datetime"].dt.floor("h")
    return weather

@asset(group_name = "raw_data")
def holidays_data(data_loader: BikeRentalDataLoader) -> pd.DataFrame:
    holidays = data_loader.load_dataset("holidays_data")
    holidays["date"] = pd.to_datetime(holidays["date"]).dt.date
    return holidays
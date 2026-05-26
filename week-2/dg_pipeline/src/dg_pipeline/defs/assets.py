import pandas as pd
from dagster import asset

@asset
def registered_rentals() -> pd.DataFrame:
    registered = pd.read_csv("../data/registered_bike_rentals.csv")
    registered["datetime"] = pd.to_datetime(registered["datetime"])
    registered["hour"] = registered["datetime"].dt.floor("h")
    return registered

@asset
def direct_pickups() -> pd.DataFrame:
    direct = pd.read_csv("../data/direct_pickup_bike_rentals.csv")
    direct["datetime"] = pd.to_datetime(direct["datetime"])
    direct["hour"] = direct["datetime"].dt.floor("h")
    return direct

@asset
def weather_data() -> pd.DataFrame:
    weather = pd.read_csv("../data/weather.csv")
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    weather["hour"] = weather["datetime"].dt.floor("h")
    return weather

@asset
def holidays_data() -> pd.DataFrame:
    holidays = pd.read_csv("../data/holidays.csv")
    holidays["date"] = pd.to_datetime(holidays["date"]).dt.date
    return holidays


# hourly_rentals


# rentals_with_weather


# final_data


# @asset
# def final_csv(final_data):
#     final_data.to_csv("../data/final.csv", index=False)
#     return "../data.final.csv"
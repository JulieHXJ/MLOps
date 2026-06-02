## Bike Rental Dagster Pipeline

This project implements a Dagster asset pipeline for transforming raw bike rental, weather, and holiday data into a model-ready location-level hourly demand dataset.

Each row in the final dataset represents the rental demand for one location during one specific hourly time window. The dataset is enriched with time-based features, hourly weather information, missing-data markers, and holiday information.

### Pipeline Overview

The pipeline contains four raw data assets and five derived assets:

- `registered_rentals`: loads registered bike rental event data.
- `direct_pickups`: loads direct pickup rental event data.
- `weather_data`: loads weather observations.
- `holidays_data`: loads holiday information.
- `all_rental_events`: combines registered rentals and direct pickups into one rental event table.
- `hourly_location_rentals`: aggregates rental events by hour, location, and rental type.
- `rentals_with_weather`: merges location-level hourly rentals with weather data and handles missing weather values.
- `final_rental_data`: merges holiday information and creates an `is_holiday` feature.
- `final_model_csv`: saves the final full and model-ready datasets as CSV files.

### Asset Dependency Graph

The pipeline is organized as a sequence of Dagster assets. The raw rental assets are first transformed into hourly rental demand, then enriched with weather and holiday information, and finally exported as a model-ready CSV file.

```text
registered_rentals ─┐
                    ├──> hourly_location_rentals ─┐
direct_pickups ─────┘                              │
                                                   ├──> rentals_with_weather ─┐
weather_data ──────────────────────────────────────┘                          │
                                                                              ├──> final_bike_rental_data ───> final_model_csv
holidays_data ────────────────────────────────────────────────────────────────┘
```


### Data Loading

The raw assets load CSV files from the data directory. Rental and weather timestamps are converted to pandas datetime values, and an hourly timestamp column is created using `dt.floor("h")`. Holiday dates are converted to Python date values so they can be merged with the hourly rental data.

### Transformations

The `all_rental_events` asset selects the hour and location_id columns from both rental sources and adds a rental_type column to distinguish registered rentals from direct pickups. These two event tables are then concatenated into one combined rental event table.

The `hourly_location_rentals` asset aggregates rental events by hour, location_id, and rental_type. This produces separate registered_count and direct_count columns for each location-hour pair. A total_count column is created by adding the two rental counts together.

To create a complete location-level hourly dataset, the pipeline builds a full hour-location table using all hourly timestamps and all observed rental locations. Missing location-hour rows are added, marked and filled with 0.

Time-based features are then created from the hourly timestamp, including `date`, `hour_of_day`, `day_of_week`, `month`, and `is_weekend`.

The `rentals_with_weather` asset joins the location-level hourly rental dataset with weather data by hour. Since weather data is hourly and not location-specific, the same weather observation is attached to all locations within the same hour. Missing weather rows are marked before filling values. Numeric weather columns are interpolated and then forward/backward filled, while categorical weather columns are filled using forward fill and backward fill.

The `final_rental_data` asset merges holiday data by date and creates a numeric `is_holiday` feature. The holiday text column is kept in this asset for validation.

The `final_model_csv` asset saves two CSV files: a full version that keeps the holiday name column for checking, and a model-ready version that removes the holiday text column and keeps only the numeric `is_holiday` feature.

### Final Dataset

The final dataset contains:

- location-level hourly rental counts
- registered_count, direct_count, and total_count
- location identifier
- time-based features
- weather features
- missing-rental and missing-weather markers
- holiday indicator

Boolean features are stored as `0` and `1` to make the CSV easier to use in later machine learning steps.

### Validation

The Dagster assets attach metadata such as row counts, count sums, missing marker counts, output paths, and dataset previews. These checks are used to verify that the hourly aggregation, weather merge, holiday merge, and CSV export were completed successfully.
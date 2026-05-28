## Bike Rental Dagster Pipeline

This project implements a Dagster asset pipeline for transforming raw bike rental, weather, and holiday data into a model-ready hourly dataset.

### Pipeline Overview

The pipeline contains four raw data assets and four derived assets:

- `registered_rentals`: loads registered bike rental event data.
- `direct_pickups`: loads direct pickup rental event data.
- `weather_data`: loads weather observations.
- `holidays_data`: loads holiday information.
- `hourly_rentals`: aggregates rental events by hour and creates time-based features.
- `rentals_with_weather`: merges hourly rentals with weather data and handles missing weather values.
- `final_bike_rental_data`: merges holiday information and creates an `is_holiday` feature.
- `final_model_csv`: saves the final model-ready dataset as a CSV file.

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

### Implementation Approach

I first implemented the transformation logic in a Jupyter notebook. This allowed me to explore the raw datasets, understand the column structure, test the table joins, check missing values, and validate the final dataset step by step.

After the notebook workflow produced the expected final dataset, I simplified the logic and migrated it into Dagster. Each major step became a separate asset: loading raw data, aggregating rentals by hour, merging weather data, adding holiday information, and exporting the final model-ready CSV. This made the workflow reproducible and easier to monitor through the Dagster UI.


### Data Loading

The raw assets load CSV files from the data directory. Rental and weather timestamps are converted to pandas datetime values, and an hourly timestamp column is created using `dt.floor("h")`. Holiday dates are converted to Python date values so they can be merged with the hourly rental data.

### Transformations

The `hourly_rentals` asset aggregates registered rentals and direct pickups by hour. It merges the two hourly count tables, creates a complete hourly index, fills missing rental counts with zero, and creates time-based features such as hour of day, weekday, and weekend indicator.

The `rentals_with_weather` asset joins the hourly rental dataset with weather data. Missing weather rows are marked before filling values. Numeric weather columns are interpolated, while categorical weather columns are filled using forward fill and backward fill.

The `final_bike_rental_data` asset merges holiday data by date and creates an `is_holiday` feature. The `final_model_csv` asset removes the holiday text column and saves the final model-ready CSV file.

### Final Dataset

The final dataset contains:

- hourly rental counts
- total rental demand
- time-based features
- weather features
- missing-rental and missing-weather markers
- holiday indicator

Boolean features are stored as `0` and `1` to make the CSV easier to use in later machine learning steps.

### Validation

The Dagster assets attach metadata such as row counts, count sums, missing marker counts, output paths, and dataset previews. These checks are used to verify that the hourly aggregation, weather merge, holiday merge, and CSV export were completed successfully.
# Air Quality Risk Predictor

An end-to-end supervised ML system that predicts air quality risk levels for
**Any city** (default — easily changeable) from weather data, with a
Streamlit app for interactive prediction and visualization.

## How the pieces fit together

1. **`fetch_weather.py`** pulls ~9-12 months of hourly weather (temperature,
   humidity, pressure, wind, precipitation) from Open-Meteo's free archive API.
2. **`fetch_air_quality.py`** finds OpenAQ monitoring stations near the city
   coordinates and pulls hourly PM2.5 / PM10 / NO2 readings for the same window.
   
   
Merging these weather info with air_quaility and then feeding it to 3 models as of now then , getting a future weather report and can predict pollution index

## Week 3 tasks

### Auckland
Merged the weather and pollution info into one

=== BEFORE: raw weather + raw air quality (separate, uncleaned) ===
Weather raw:        8664 rows  | columns: ['datetime', 'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m', 'precipitation']
Air quality raw:    4026 rows  | columns: ['datetime', 'pm25', 'source']

Missing values in raw air quality data:
datetime    0
pm25        0
source      0
dtype: int64

=== RUNNING merge_clean.py ===
Weather rows: 8664 | AQ rows: 8663 | Merged (labeled) rows: 4083

Risk category distribution:
risk_category
Good    4083
Name: count, dtype: int64

Saved merged & cleaned dataset to \data\processed\auckland\merged_clean.csv

=== AFTER: merged, cleaned, and labeled ===
Final merged dataset: 4083 rows
Columns: ['datetime', 'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m', 'precipitation', 'pm25', 'source', 'aqi', 'risk_category']

Missing values after cleaning:
temperature_2m    0
pm25              0
aqi               0
dtype: int64

Note: this city's station(s) only report ['pm25'] -- no ['pm10', 'no2'] sensor was found nearby. AQI is still computed correctly (max of whichever sub-indices are available).

Sample rows:
           datetime  temperature_2m   pm25      aqi risk_category
2025-07-31 00:00:00            11.7 0.7855 3.272917          Good
2025-07-31 01:00:00            11.5 0.5540 2.308333          Good
2025-07-31 02:00:00            11.2 0.8690 3.620833          Good
2025-07-31 03:00:00            10.9 1.0070 4.195833          Good
2025-07-31 04:00:00            10.6 0.6365 2.652083          Good

Done -- merged_clean.csv saved to data/processed/auckland/


*For auckland it will be mostly good*

### Delhi

=== BEFORE: raw weather + raw air quality (separate, uncleaned) ===
Weather raw:        8664 rows  | columns: ['datetime', 'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m', 'precipitation']
Air quality raw:    5127 rows  | columns: ['datetime', 'no2', 'pm10', 'pm25', 'source']

Missing values in raw air quality data:
datetime      0
no2         228
pm10        298
pm25         70
source        0
dtype: int64

=== RUNNING merge_clean.py ===
Weather rows: 8664 | AQ rows: 8664 | Merged (labeled) rows: 5230

Risk category distribution:
risk_category
Unhealthy                       2089
Unhealthy (Sensitive Groups)    1197
Moderate                        1135
Very Unhealthy                   532
Hazardous                        213
Good                              64
Name: count, dtype: int64

Saved merged & cleaned dataset to \data\processed\delhi\merged_clean.csv

=== AFTER: merged, cleaned, and labeled ===
Final merged dataset: 5230 rows
Columns: ['datetime', 'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m', 'precipitation', 'pm25', 'pm10', 'no2', 'source', 'aqi', 'risk_category']

Missing values after cleaning:
temperature_2m      0
pm25                0
pm10              144
no2                95
aqi                 0
dtype: int64

Sample rows:
           datetime  temperature_2m  pm25        aqi                risk_category
2025-07-31 00:00:00            26.3  50.0 136.703518 Unhealthy (Sensitive Groups)
2025-07-31 01:00:00            26.2  48.3 132.517588 Unhealthy (Sensitive Groups)
2025-07-31 02:00:00            25.8  43.0 119.467337 Unhealthy (Sensitive Groups)
2025-07-31 03:00:00            25.5  23.0  73.922747                     Moderate
2025-07-31 04:00:00            25.3  38.5 108.386935 Unhealthy (Sensitive Groups)

Done -- merged_clean.csv saved to data/processed/delhi/



## Week 4 tasks

### Auckland 
=== BEFORE: merged_clean.csv (week 3's output) ===
Rows: 4083  |  Columns: 11
Column names: ['datetime', 'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m', 'precipitation', 'pm25', 'source', 'aqi', 'risk_category']

=== RUNNING feature_engineering.py ===
Built 57 columns x 4042 rows -> \data\processed\auckland\features.csv

=== AFTER: features.csv ===
Rows: 4042  |  Columns: 57

46 new engineered feature columns added, e.g.:
  - is_weekend
  - hour_sin
  - hour_cos
  - temperature_2m_roll_24h_mean
  - temperature_2m_roll_24h_std
  - relative_humidity_2m_roll_24h_mean
  - relative_humidity_2m_roll_24h_std
  - surface_pressure_roll_24h_mean

=== Sample rows showing a few engineered features ===
           datetime risk_category  hour  is_weekend  pm25_roll_24h_mean  pm25_lag_24h
2025-08-01 17:00:00          Good    17           0            3.242420         3.120
2025-08-01 18:00:00          Good    18           0            3.218878         2.790
2025-08-01 19:00:00          Good    19           0            3.235336         2.700
2025-08-01 20:00:00          Good    20           0            3.354920         1.465
2025-08-01 21:00:00          Good    21           0            3.487836         1.675

Note: 4083 -> 4042 rows -- the drop is expected: rolling/lag features need prior history, so the first ~7 days of the series (before a full week of lookback exists) are dropped.

Done -- features.csv saved to data/processed/auckland/

### Delhi

=== BEFORE: merged_clean.csv (week 3's output) ===
Rows: 5230  |  Columns: 13
Column names: ['datetime', 'temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m', 'precipitation', 'pm25', 'pm10', 'no2', 'source', 'aqi', 'risk_category']

=== RUNNING feature_engineering.py ===
Built 69 columns x 5007 rows -> data\processed\delhi\features.csv

=== AFTER: features.csv ===
Rows: 5007  |  Columns: 69

56 new engineered feature columns added, e.g.:
  - is_weekend
  - hour_sin
  - hour_cos
  - temperature_2m_roll_24h_mean
  - temperature_2m_roll_24h_std
  - relative_humidity_2m_roll_24h_mean
  - relative_humidity_2m_roll_24h_std
  - surface_pressure_roll_24h_mean

=== Sample rows showing a few engineered features ===
           datetime risk_category  hour  is_weekend  pm25_roll_24h_mean  pm25_lag_24h
2025-08-01 19:00:00      Moderate    19           0           26.317708         17.25
2025-08-01 20:00:00      Moderate    20           0           26.755208         18.00
2025-08-01 21:00:00      Moderate    21           0           26.130208         30.00
2025-08-01 22:00:00      Moderate    22           0           25.038542         42.00
2025-08-01 23:00:00      Moderate    23           0           23.976042         42.00

Note: 5230 -> 5007 rows -- the drop is expected: rolling/lag features need prior history, so the first ~7 days of the series (before a full week of lookback exists) are dropped.

Done -- features.csv saved to data/processed/delhi/

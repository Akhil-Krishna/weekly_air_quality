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
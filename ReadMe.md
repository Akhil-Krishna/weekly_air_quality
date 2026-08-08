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
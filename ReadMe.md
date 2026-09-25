# Air Quality Risk Predictor

An end-to-end supervised machine learning system that predicts air quality
risk levels from weather data, with a Streamlit app for interactive
prediction, model explainability, and cross-city comparison.

Built for two cities so far: **Delhi** (high pollution variance, fully
trained) and **Auckland** (clean-air baseline, no model trained -- and that
absence is itself a real, explained finding, not a gap).

---

## What this project actually does

1. Collects real hourly weather (Open-Meteo) and real air pollution readings
   (OpenAQ ground stations, with an automatic fallback to Open-Meteo's
   model-based Air Quality API for cities with no nearby station).
2. Converts raw pollutant concentrations into an AQI value and a risk
   category (Good / Moderate / Unhealthy for Sensitive Groups / Unhealthy /
   Very Unhealthy / Hazardous), using the official US EPA breakpoint tables.
3. Cleans and merges the two datasets, engineers time-based and
   rolling/lag features, and trains three models (Logistic Regression,
   Random Forest, XGBoost) on a **chronological** train/test split so the
   model is evaluated the way it would actually be used -- predicting the
   future from the past.
4. Serves everything through a Streamlit app: historical lookup, live
   forecast prediction, model explainability, a Delhi-vs-Auckland
   comparison, and a methodology comparison against MetService's pollen
   forecasting.

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free OpenAQ API key

OpenAQ's v3 API requires a free key (Open-Meteo needs no key at all).

1. Register at https://explore.openaq.org/register
2. Set it as an environment variable, or copy `.env.example` to `.env` and

### 3. Run the pipeline for a city

`config.py` controls which city gets collected/trained. The first three
lines are the only thing you normally need to change:

```python
CITY_NAME = "Delhi"
LATITUDE = 28.6139
LONGITUDE = 77.2090
```

Then run everything end to end:

```bash
python run_pipeline.py
```

This fetches weather, fetches air quality (OpenAQ first, falling back to
Open-Meteo Air Quality automatically if no station is nearby), merges and
cleans the data, engineers features, trains and compares all three models,
and saves the best one. Every city gets its own folder automatically
(`data/raw/<city>/`, `data/processed/<city>/`, `models/<city>/`) so running
this for a second city never overwrites the first.

To add a city to the app's city switcher, also add it to the `CITIES` dict
in `config.py`:

```python
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Auckland": {"lat": -36.8485, "lon": 174.7633},
}
```

### 4. Launch the app

```bash
streamlit run app/streamlit_app.py
```

---

## The app

Five tabs, with a city selector at the top that switches between any city
that's been through the pipeline:

| Tab | What it does |
|---|---|
| **Historical View** | Pick a past date/hour, see the actual weather, actual pollution reading, and the model's prediction next to what really happened. |
| **Forecast View** | Pulls live upcoming weather from Open-Meteo and predicts risk up to 7 days ahead -- a genuine decision-support tool, not just a dashboard of the past. |
| **Explainability** | Shows which features actually drive the model's predictions (feature importance on the winning model).Per-prediction SHAP breakdowns -- why the model predicted a specific category for one specific date/hour -- are available as an expander inside the Historical View tab. |
| **City Comparison** | Delhi vs. Auckland, using real computed statistics from each city's own data -- not illustrative numbers. |
| **Allergy Comparison** | How this project's approach compares to MetService's pollen/allergy forecasting, including an honest explanation of why a live data comparison isn't possible. |

A city with data but no trained model (currently Auckland) doesn't break
the app -- each tab explains why, using that city's real numbers, instead
of crashing or hiding the city entirely.

---

## Why Auckland has no trained model

This isn't a bug or missing work -- it's a real result. Auckland's entire
collected dataset (4083 hourly readings, ~11 months) fell into a single
risk category: **Good**, 100% of the time. Mean AQI across the whole period
was **7.9**, against Delhi's **153.7** over the same window.

A classification model needs at least two classes to learn to tell apart.
With only one present, there's nothing to distinguish -- so `train_models.py`
detects this upfront and skips training with a clear explanation, rather
than crashing (which is what scikit-learn does by default on single-class
data) or fitting a model that trivially always predicts the same thing.

This is the actual point of running two cities side by side: one with
enough pollution variance to make a real prediction problem, and one clean
enough that the "problem" itself disappears. That contrast is presented
directly in the app's City Comparison tab.

---

## Real results (Delhi)

Chronological split: last 3 months held out as the test set, first ~9
months for training.

| Model | F1 (macro) | Precision (macro) | Recall (macro) |
|---|---|---|---|
| Logistic Regression | 0.171 | 0.196 | 0.209 |
| Random Forest | 0.201 | 0.248 | 0.225 |
| **XGBoost (best)** | **0.221** | 0.236 | 0.228 |

These F1 scores are modest, and that's honest, not a bug to hide: weather
alone is a real but partial predictor of pollution -- local traffic,
industrial activity, and construction (none of which this project collects)
also drive Delhi's air quality. Worth stating plainly if asked, rather than
implying weather alone should predict this well.

**AQI calculation cross-check:** our own EPA-breakpoint AQI math was
validated against Open-Meteo's independently-computed `us_aqi` field over a
360-hour sample: mean AQI 176.4 (ours) vs. 175.0 (Open-Meteo's),
correlation 0.574. Close agreement on the mean, moderate correlation --
reasonable given the two sources use different pollutant sourcing and
spatial resolution, and is treated as supporting evidence the labeling
logic is implemented correctly, not as certainty.

---

## Project structure

```
weekly_air_quality-master/
├── config.py                    # city settings, API endpoints, file paths
├── run_pipeline.py              # runs the full pipeline end to end
├── requirements.txt
├── .env.example
├── src/
│   ├── fetch_weather.py            
│   ├── fetch_air_quality.py        
│   ├── fetch_air_quality_openmeteo.py 
│   ├── labeling.py                  
│   ├── merge_clean.py              
│   ├── feature_engineering.py      
│   ├── train_models.py             
│   ├── explainability.py            # feature importance
│   ├── validate_aqi_crosscheck.py   # validates labeling.py against Open-Meteo
│   └── utils.py                    
├── app/
│   └── streamlit_app.py        
├── data/{raw,processed}/<city>/ # per-city data, kept separate automatically
├── models/<city>/               # per-city trained model + metrics
└── demo_week4.py / demo_week5.py / demo_week6.py   # weekly progress demos
```

---

## Design notes and honest limitations

- **Chronological split, not random.** The model is tested on the most
  recent months only, held out entirely from training -- this mirrors how
  it would actually be used (predicting the future from the past) and
  avoids the leakage a random shuffle-split would introduce.
- **Macro F1 over accuracy.** Risk categories are heavily imbalanced
  (Delhi is mostly Unhealthy/Moderate; Auckland is entirely Good) --
  accuracy would hide poor performance on rarer classes. `class_weight="balanced"`
  is used where supported.
- **Hybrid air quality source, disclosed per row.** Every row carries a
  `source` column (`openaq_ground` or `openmeteo_model`) recording which
  data source was actually used -- ground-truth sensor data and
  model-estimated data are genuinely different in kind, and this is never
  hidden.
- **Forecast view assumption.** Future weather comes from a real forecast,
  but future pollution is exactly what's being predicted -- so the
  forecast's rolling/lag pollutant features carry forward the most recent
  known real values rather than inventing numbers. Disclosed in the app
  itself.
- **Allergy comparison is methodology-level, not live data.** Checked
  directly before building this: Open-Meteo's pollen data only covers
  Europe (not NZ or India), and MetService's real-time pollen feed is a
  licensed commercial API, not a public one. Rather than ship a fragile,
  untested scraper against a page that loads its data client-side, the
  comparison stays honest about what it is.

---



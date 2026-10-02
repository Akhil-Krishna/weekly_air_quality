# Air Quality Risk Predictor — Complete Project Report

A full walkthrough of what this project is, how the data flows through it, how
each part was built, and why the design decisions were made. Every section
gives the plain-English version first and the technical detail second.

**Version:** post-v2.0.2, October 2026
**Codebase:** ~4,220 lines of Python across 19 files, plus 69 tests

---

## Table of contents

1. [What this project is, in one page](#1-what-this-project-is-in-one-page)
2. [The problem, and why it is harder than it looks](#2-the-problem-and-why-it-is-harder-than-it-looks)
3. [Technology stack](#3-technology-stack)
4. [Architecture at a glance](#4-architecture-at-a-glance)
5. [Data sources and APIs](#5-data-sources-and-apis)
6. [The pipeline, stage by stage](#6-the-pipeline-stage-by-stage)
7. [The science: AQI and risk categories](#7-the-science-aqi-and-risk-categories)
8. [Feature engineering in detail](#8-feature-engineering-in-detail)
9. [The machine learning design](#9-the-machine-learning-design)
10. [The three models](#10-the-three-models)
11. [The Streamlit app, tab by tab](#11-the-streamlit-app-tab-by-tab)
12. [Explainability](#12-explainability)
13. [Validation and testing](#13-validation-and-testing)
14. [Results](#14-results)
15. [Engineering decisions and bugs fixed](#15-engineering-decisions-and-bugs-fixed)
16. [Deployment and operations](#16-deployment-and-operations)
17. [Limitations and honest caveats](#17-limitations-and-honest-caveats)
18. [File-by-file map](#18-file-by-file-map)
19. [Glossary](#19-glossary)
20. [Suggested presentation outline](#20-suggested-presentation-outline)

---

## 1. What this project is, in one page

### In plain terms

This is a system that **predicts how bad the air will be, using the weather**.

It collects a year of real hourly weather (temperature, humidity, wind,
pressure, rainfall) and a year of real hourly pollution readings (the stuff in
the air: PM2.5, PM10, NO2) for six cities. It converts the pollution readings
into the familiar Air Quality Index, buckets that into six risk levels from
"Good" to "Hazardous", and then trains machine-learning models to learn the
relationship between *weather conditions* and *air quality risk*.

Once trained, the system can take a live weather forecast and say: "given this
upcoming weather, the air in Delhi will probably be Unhealthy on Thursday
afternoon."

All of it is wrapped in a web app where you can look up any past hour, see a
live forecast, understand *why* the model made a prediction, compare cities
against each other, and check a pollen forecast alongside.

### In technical terms

A supervised multi-class classification pipeline:

```
Open-Meteo hourly weather  ─┐
                            ├─► UTC join ─► EPA AQI ─► 6-class label
OpenAQ hourly pollutants   ─┘
                                    │
                                    ▼
                    time + rolling + lag feature engineering (64 features)
                                    │
                                    ▼
                    chronological train/test split (last 3 months held out)
                                    │
                                    ▼
          LogisticRegression  vs  RandomForest  vs  XGBoost
                                    │
                                    ▼
                    best-by-macro-F1 model persisted per city
                                    │
                                    ▼
                         5-tab Streamlit application
```

### The key insight that makes it a real project

Weather does not *cause* all pollution — traffic, industry and construction do
too, and none of those are measured here. So the model is deliberately solving
a **partially-observable** problem, and the honest reporting of that limitation
is part of the work. A model that claimed 95% accuracy here would be lying or
leaking.

---

## 2. The problem, and why it is harder than it looks

### The naive view

"Take weather, predict pollution. It's just a classifier."

### What actually makes it hard

| Challenge | Why it bites |
|---|---|
| **Two clocks** | Weather APIs return local time; pollution APIs return UTC. Joining them naively misaligns every label from its features by hours. |
| **Target leakage** | The label is computed *from* the pollutants. Any feature derived from current-hour pollutants silently contains the answer. |
| **Gappy timelines** | Sensors go offline. A "24-hour average" computed over rows rather than hours can silently span weeks. |
| **Severe class imbalance** | Delhi is mostly Unhealthy; Auckland is 96% Good. Accuracy is a useless metric here. |
| **Time-ordered data** | A random train/test split lets the model see the future. The split must be chronological. |
| **Unreliable sources** | Stations die without announcement. Metadata lies. Pagination APIs misreport totals. |
| **Partial observability** | Weather explains some of the variance, never all of it. |

Most of the engineering in this project exists to handle these seven problems
honestly, rather than to squeeze out another point of F1.

---

## 3. Technology stack

### Plain terms

Python for everything. Standard data-science libraries for the maths. Streamlit
to turn the results into a website without writing any HTML, CSS or JavaScript.
All data comes from free public web APIs.

### Technical detail

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.11 | Entire codebase |
| **Data handling** | pandas 3.0, numpy | Dataframes, time series, resampling |
| **ML** | scikit-learn 1.9 | LogisticRegression, RandomForest, StandardScaler, metrics |
| **ML** | XGBoost 3.2 | Gradient-boosted trees |
| **Explainability** | SHAP 0.51 | Per-prediction attribution |
| **Persistence** | joblib | Model/scaler/encoder serialisation |
| **Web UI** | Streamlit 1.60 | Entire front end |
| **Charts** | Plotly | Interactive charts inside Streamlit |
| **HTTP** | requests | API calls with retry/backoff |
| **Config** | python-dotenv | Loads API keys from `.env` |
| **Testing** | pytest | 69 regression tests |

**Why Streamlit?** It renders a Python script top-to-bottom as a web page. No
front-end code, no API layer, no separate server. The trade-off is that the
whole script re-executes on every interaction, which is why caching
(`@st.cache_data`, `@st.cache_resource`) matters so much here.

---

## 4. Architecture at a glance

### The whole system

```
┌──────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                        │
│  Open-Meteo Archive   Open-Meteo Forecast   Open-Meteo AQ   OpenAQ   │
│  Atmospore (pollen)                                                   │
└──────────────┬───────────────────────────────────────────────────────┘
               │ HTTPS + retry/backoff (src/utils.py)
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      COLLECTION  (per city, UTC)                     │
│  fetch_weather.py          →  data/raw/<city>/weather_raw.csv        │
│  fetch_air_quality.py      →  data/raw/<city>/air_quality_raw.csv    │
│    └─ quality gate: coverage + staleness → Open-Meteo fallback       │
└──────────────┬───────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      PROCESSING                                      │
│  labeling.py       EPA breakpoints → AQI → 6 risk categories         │
│  merge_clean.py    UTC inner join, sanity-clean, coverage report     │
│                    →  data/processed/<city>/merged_clean.csv         │
│  feature_engineering.py   time + rolling + lag  (64 features)        │
│                    →  data/processed/<city>/features.csv             │
└──────────────┬───────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      MODELLING                                       │
│  train_models.py   chronological split → 3 models → best by macro F1 │
│    → models/<city>/{best_model, scaler, label_encoder,               │
│                     feature_list, model_comparison}                  │
└──────────────┬───────────────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION                                    │
│  app/streamlit_app.py — 5 tabs, city switcher                        │
│  Historical | Forecast | Explainability | City Comparison | Allergy  │
└──────────────────────────────────────────────────────────────────────┘
```

### Per-city isolation

Every artefact lives under a city-specific folder, derived once at import time
in `config.py`:

```
data/raw/delhi/        data/processed/delhi/        models/delhi/
data/raw/auckland/     data/processed/auckland/     models/auckland/
...
```

This means running the pipeline for one city can never overwrite another's
data. The city is chosen by the `AQ_CITY` environment variable, which
`run_pipeline.py` sets per subprocess.

### Why subprocesses

`config.py` resolves the target city **and all its file paths at import time**.
A single process therefore cannot switch cities midway without stale paths
leaking. `run_pipeline.py --all` spawns one subprocess per city with `AQ_CITY`
set, which also means one city failing does not take down the rest.

---

## 5. Data sources and APIs

### Plain terms

Five free web services. Two give weather, two give pollution, one gives pollen.
Only two of them need a key, and both keys are free.

### Technical detail

#### 5.1 Open-Meteo Archive API — historical weather

- **URL:** `https://archive-api.open-meteo.com/v1/archive`
- **Auth:** none
- **Used for:** the model's input features — a year of hourly weather
- **Variables pulled:** `temperature_2m`, `relative_humidity_2m`,
  `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`
- **Chunking:** 90-day windows, to stay polite and avoid oversized responses
- **Critical parameter:** `timezone=UTC` (see §15.2)

#### 5.2 Open-Meteo Forecast API — upcoming weather

- **URL:** `https://api.open-meteo.com/v1/forecast`
- **Auth:** none
- **Used for:** the Forecast tab — up to 7 days ahead
- **Key parameter:** `past_days=8` — returns the 8 days *before* now in the
  same response, so rolling 24h/7d features at the start of the forecast
  horizon are computed over one genuinely contiguous series

#### 5.3 Open-Meteo Air Quality API — fallback pollution + validation

- **URL:** `https://air-quality-api.open-meteo.com/v1/air-quality`
- **Auth:** none
- **Used for:** two distinct jobs
  1. **Fallback source** when a city has no usable ground station
  2. **Independent validation** of our own AQI maths via its `us_aqi` field
- **Variables:** `pm2_5`, `pm10`, `nitrogen_dioxide`, `us_aqi`
- **Nature:** model/satellite-derived (CAMS reanalysis), *not* a physical sensor

#### 5.4 OpenAQ v3 API — ground-station pollution

- **URL:** `https://api.openaq.org/v3`
- **Auth:** free API key, `X-API-Key` header
- **Used for:** the ground truth the labels are built from
- **Three-step flow:**
  1. `GET /locations?coordinates=lat,lon&radius=25000` — find stations
  2. Read each location's embedded `sensors[]` list — one sensor per pollutant
  3. `GET /sensors/{id}/hours` — paginated hourly measurements
- **Rate limiting:** 1.2 s delay between requests, plus exponential backoff
  honouring `Retry-After` on HTTP 429
- **Station selection:** nearest 6 **that are still reporting** (see §15.5)

#### 5.5 Atmospore Pollen API — allergy tab

- **URL:** `https://pollenapi.com/v1/pollen`
- **Auth:** free API key, `x-api-key` header
- **Budget:** free tier is 100 calls/day **shared across all cities**; the code
  self-limits to 90 and keeps a ledger plus a daily cache so the app degrades
  to cached data instead of erroring
- **Independent of the ML pipeline** — this tab works even if no model exists

#### 5.6 Shared HTTP behaviour (`src/utils.py`)

```python
get_json_with_retries(url, params, headers, max_retries=6, backoff=3.0)
```

- **429 (rate limited):** exponential backoff 3→6→12→24→48→96 s, capped at
  120 s, honouring the server's `Retry-After` header
- **Other 4xx:** fails fast and surfaces the response body — a 400/401/422 is a
  malformed or unauthorised request, and retrying identical bad parameters six
  times just wastes a minute before failing anyway
- **Network errors:** linear backoff retry

---

## 6. The pipeline, stage by stage

Run it with:

```bash
python run_pipeline.py --city Delhi     # one city
python run_pipeline.py --all            # all six
python run_pipeline.py --skip-fetch     # re-derive from existing raw CSVs
```

### Stage 1 — Fetch weather (`src/fetch_weather.py`)

**Plain:** downloads a year of hourly weather and saves it as a CSV.

**Technical:**
- Iterates 90-day chunks from `START_DATE` to `END_DATE`
- Requests `timezone=UTC` so timestamps share a clock with OpenAQ
- Concatenates chunks, de-duplicates on `datetime`, sorts
- Output: `data/raw/<city>/weather_raw.csv`, ~8,664 rows

### Stage 2 — Fetch air quality (`src/fetch_air_quality.py`)

**Plain:** finds pollution sensors near the city, downloads their readings, and
— crucially — checks whether what came back is actually good enough to use. If
not, it falls back to the satellite-model source.

**Technical:**

```
find_nearby_locations(lat, lon, radius, max_locations=6, active_since=START_DATE)
   │  ├─ drop stations whose datetimeLast < START_DATE   (dead sensors)
   │  ├─ rank confirmed-live by distance, then undated ones
   │  └─ take nearest 6, printing each with its last-report date
   ▼
for each sensor: fetch_sensor_measurements()  ← paginated, 1000/page
   ▼
average across stations per (timestamp, pollutant) → pivot wide
   ▼
assess_ground_coverage(df, start, end)
   ├─ coverage ≥ 50% of requested hours?          AND
   └─ last reading within 30 days of END_DATE?
        ├─ yes → keep OpenAQ ground data
        └─ no  → fall back to Open-Meteo Air Quality
```

The quality gate matters because "is there a station nearby" and "is there
usable data" are different questions — Hamilton has two stations within 3 km
that both stopped reporting on 2025-12-03.

Output: `data/raw/<city>/air_quality_raw.csv` with a `source` column recording
`openaq_ground` or `openmeteo_model` **per row**.

### Stage 3 — Merge and clean (`src/merge_clean.py`)

**Plain:** lines the weather up against the pollution hour by hour, removes
impossible readings, works out the AQI and risk label, and reports how much of
the year actually survived.

**Technical:**
1. **Clean weather** — coerce numerics, resample to a strict hourly grid,
   linear-interpolate gaps of at most 6 hours
2. **Clean pollutants** — null out negatives (a known sensor failure mode) and
   anything above the physical sanity ceilings in
   `config.POLLUTANT_SANITY_CEILINGS` (PM2.5 2000, PM10 3000, NO2 1000 µg/m³)
3. **Inner join on `datetime`** — valid because both sides are UTC
4. **Label** via `labeling.label_dataframe` → adds `aqi` and `risk_category`
5. **Coverage report** — prints hours present vs hours expected, warns below 85%

Output: `data/processed/<city>/merged_clean.csv`

> **Design note.** Outlier handling deliberately does *not* use IQR winsorizing.
> It previously clipped at `q3 + 3×IQR` computed over train and test together,
> which both leaked test statistics into cleaning and removed exactly the
> extreme readings the Very Unhealthy and Hazardous classes are made of. For a
> heavy-tailed quantity like Delhi PM2.5, being far above the 75th percentile is
> a real pollution event, not an error.

### Stage 4 — Feature engineering (`src/feature_engineering.py`)

Covered fully in §8.

Output: `data/processed/<city>/features.csv`, 64 model features + metadata

### Stage 5 — Train and compare (`src/train_models.py`)

Covered fully in §9 and §10.

Output: five files in `models/<city>/`.

---

## 7. The science: AQI and risk categories

### Plain terms

Pollution is measured in micrograms per cubic metre — a number that means
nothing to most people. The Air Quality Index converts those concentrations
onto a single 0–500 scale where everyone understands that 50 is fine and 300
is dangerous. We then bucket that scale into six named risk levels.

### Technical detail (`src/labeling.py`)

#### Step 1 — Per-pollutant sub-index

Each pollutant has an EPA breakpoint table mapping concentration ranges to AQI
ranges. Within a band the mapping is linear:

```
AQI = aqi_low + (aqi_high − aqi_low) / (conc_high − conc_low) × (conc − conc_low)
```

**PM2.5, EPA 2024 revision** (the current standard, selectable via
`config.PM25_BREAKPOINT_VERSION`):

| Concentration (µg/m³) | AQI |
|---|---|
| 0.0 – 9.0 | 0 – 50 |
| 9.1 – 35.4 | 51 – 100 |
| 35.5 – 55.4 | 101 – 150 |
| 55.5 – 125.4 | 151 – 200 |
| 125.5 – 225.4 | 201 – 300 |
| 225.5 – 325.4 | 301 – 500 |

PM10 uses integer µg/m³ bands; NO2 uses parts-per-billion bands, so OpenAQ's
µg/m³ readings are converted with the standard factor `1 ppb ≈ 1.88 µg/m³`.

#### Step 2 — The truncation rule that prevents gaps

Notice the tables have **gaps**: one band ends at 9.0, the next starts at 9.1.
A reading of 9.05 matches neither. EPA's own procedure resolves this by
**truncating the concentration to the table's precision first** (1 decimal for
PM2.5, integer for PM10 and NO2), which makes the gaps unreachable.

This is step 1 of the official method, and skipping it meant such readings
returned `NaN`, silently dropped that pollutant out of the overall maximum, and
understated the hour's AQI whenever that pollutant was the driver.

#### Step 3 — Overall AQI

```python
AQI_hour = max(sub_index(PM2.5), sub_index(PM10), sub_index(NO2))
```

The maximum, not the average — this mirrors how real AQI is reported. The
worst pollutant defines the risk.

#### Step 4 — Risk category

| AQI | Category |
|---|---|
| 0 – 50 | Good |
| 51 – 100 | Moderate |
| 101 – 150 | Unhealthy (Sensitive Groups) |
| 151 – 200 | Unhealthy |
| 201 – 300 | Very Unhealthy |
| 301+ | Hazardous |

Implemented as inclusive **upper bounds** rather than EPA's printed `(51, 100)`
ranges, because those ranges have the same gap problem — an AQI of 50.5 matched
nothing and fell through to a bare `"Hazardous"` default, labelling a near-Good
reading as the worst category on the scale.

#### Step 5 — Severity-ordered label encoding

`OrderedLabelEncoder` assigns codes in **severity order**:

```
Good=0, Moderate=1, Unhealthy(SG)=2, Unhealthy=3, Very Unhealthy=4, Hazardous=5
```

scikit-learn's `LabelEncoder` sorts alphabetically, which would give
`Good=0, Hazardous=1, Moderate=2…` — harmless to a nominal classifier, but it
puts Hazardous between Good and Moderate on every confusion-matrix axis and in
every saved report. It also fits over **all six** classes regardless of which
appear in a given city, so code 3 means the same thing everywhere.

---

## 8. Feature engineering in detail

### Plain terms

The model cannot learn from a raw timestamp. We turn each hour into 64 numbers
describing the weather now, the weather recently, the time of day and year, and
what pollution was doing a day ago.

### The 64 features

| Group | Count | Examples |
|---|---|---|
| Weather, current hour | 6 | `temperature_2m`, `wind_speed_10m`, `precipitation` |
| Weather rolling | 24 | `temperature_2m_roll_24h_mean`, `wind_speed_10m_roll_7d_std` |
| Weather lag | 6 | `temperature_2m_lag_24h` |
| Pollutant rolling | 12 | `pm25_roll_24h_mean`, `no2_roll_7d_std` |
| Pollutant lag | 3 | `pm25_lag_24h`, `pm10_lag_24h`, `no2_lag_24h` |
| Cyclical time | 5 | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend` |
| Season dummies | 4 | `season_winter`, `season_summer` … |
| Time-of-day dummies | 4 | `time_of_day_morning`, `time_of_day_night` … |

Deliberately **excluded** from the feature set: `pm25`, `pm10`, `no2`, `aqi`
(the label's own ingredients), `datetime`, `datetime_local`, and the raw
`hour` / `day_of_week` / `month` (superseded by cyclical encodings).

### Three decisions that define this module

#### 8.1 Cyclical encoding of time

Hour 23 and hour 0 are adjacent in reality but 23 apart numerically. Encoding
as a sine/cosine pair places them next to each other on a circle:

```python
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

Both are needed — one alone is ambiguous (sin is equal at 2am and 10am).

#### 8.2 Windows measured in hours, not rows

The timeline has gaps. `rolling(window=24)` over raw rows averages whatever 24
records happen to sit next to each other — in the worst observed case spanning
**46 days**, under a column named `pm25_roll_24h_mean`.

Everything is therefore computed on a **gap-filled hourly grid**: the series is
reindexed to a complete hourly `DatetimeIndex` so missing hours become
all-NaN rows. Once every row is exactly one hour after the last, a window of
N rows *is* a window of N hours, and `min_periods` can reject windows that are
mostly hole. Rows with incomplete history are then dropped rather than given a
fabricated average.

#### 8.3 Lagging the pollutant features to prevent leakage

**This is the single most important correctness decision in the project.**

`risk_category` is computed from the current hour's PM2.5/PM10/NO2. A
24-hour rolling mean that *includes* the current hour therefore contains
1/24th of the answer. A 7-day mean contains 1/168th.

So pollutant-derived rolling features are shifted back one hour:

```python
add_rolling_features(df, pollutant_cols, shift_hours=1)   # excludes current hour
add_rolling_features(df, weather_cols,   shift_hours=0)   # weather may use it
```

Weather is **exogenous** — it is an input, not derived from the label — so it
may legitimately use the current hour. Pollutants may not.

Excluding the raw pollutant columns from the feature list is *not sufficient*
on its own. That was the original bug: the raw columns were excluded, but
averages of them that included the present moment were not.

#### 8.4 Local-clock features from a UTC key

The join key is UTC. But "rush hour" and "winter" are local concepts. So:

```python
datetime        = UTC          ← the join key, never rewritten
datetime_local  = UTC → city's IANA timezone    ← hour, day, season derived here
```

Going through a real IANA zone (`Asia/Kolkata`, `Pacific/Auckland`) rather than
a fixed offset is what makes **DST** correct — Auckland is UTC+13 in January
and UTC+12 in June.

Seasons also flip below the equator: `month_to_season(1)` is `winter` in Delhi
and `summer` in Auckland.

---

## 9. The machine learning design

### 9.1 Chronological split, not random

**Plain:** we train on the older data and test on the newest three months,
because that is how the model would really be used — predicting the future
from the past.

**Technical:** a random shuffle-split would let the model train on 3pm Tuesday
and test on 2pm Tuesday — adjacent hours are near-identical, so the score would
be inflated by leakage.

The split point is derived from the **data**, not the clock:

```python
split_date = data.datetime.max() − 90 days
```

and persisted into `model_comparison.json` alongside train/test row counts and
date ranges, so the app always reports the split the saved model was genuinely
evaluated on.

Current Delhi split: train 6,079 rows → test 2,089 rows, boundary 2026-06-28.

### 9.2 Macro F1, reported two ways

**Plain:** accuracy is misleading when one class dominates. A model that always
says "Good" scores 96% accuracy in Auckland while being useless.

**Technical:** macro F1 averages the per-class F1 equally, so a rare class
matters as much as a common one. The report gives both:

| Metric | Meaning |
|---|---|
| `f1_macro` | averaged over classes **present in the test set** — the headline |
| `f1_macro_all_labels` | averaged over all six — depressed by any class with zero test rows |

A class with zero test rows contributes a guaranteed 0.0 to the mean, which is
a property of the split, not of the model. Auckland illustrates the gap
starkly: 0.523 over the 2 classes present, 0.174 over all six.

Accuracy is also stored, for context.

### 9.3 Class imbalance handling

`class_weight="balanced"` on LogisticRegression and RandomForest, which weights
each class inversely to its frequency.

### 9.4 Scaling

`StandardScaler` fitted on **train only**, then applied to test. Fitting on the
full dataset would leak test distribution statistics into training.

Trees don't need scaling; LogisticRegression does. One scaler is used for all
three so the saved artefact is model-agnostic.

### 9.5 The single-class guard

A classifier needs at least two classes. `train_models.py` detects a
single-class dataset upfront and skips training with an explanation, rather
than crashing (scikit-learn's default) or fitting a model that trivially always
predicts one thing. It also guards the narrower case where the *train split*
alone is single-class.

---

## 10. The three models

### Why three

Comparing a linear model, a bagged ensemble and a boosted ensemble shows
whether the problem needs non-linearity, and gives a defensible basis for the
choice rather than reaching for XGBoost by reflex.

### 10.1 Logistic Regression

```python
LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
```

- **Plain:** draws straight dividing lines between risk categories.
- **Technical:** linear decision boundaries in 64-dimensional scaled space.
  Fast, fully interpretable via coefficients, and a genuine baseline — if it
  matched the ensembles, the extra complexity would not be justified.

### 10.2 Random Forest

```python
RandomForestClassifier(n_estimators=300, max_depth=None,
                       class_weight="balanced", random_state=42, n_jobs=-1)
```

- **Plain:** 300 decision trees vote; each sees a random slice of the data.
- **Technical:** bagging reduces variance. Captures non-linear interactions
  (e.g. low wind *and* low temperature together) without manual feature
  crosses. Unpruned trees, with averaging controlling overfitting.

### 10.3 XGBoost — usually the winner

```python
XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8,
              eval_metric="mlogloss", random_state=42, n_jobs=-1)
```

- **Plain:** builds trees one at a time, each correcting the previous one's
  mistakes.
- **Technical:** gradient boosting. `max_depth=6` and `learning_rate=0.05`
  trade depth for a slow, regularised fit; `subsample`/`colsample_bytree` at
  0.8 add stochastic regularisation.

### 10.4 The XGBoost label-space wrapper

`src/model_wrappers.py` — `ContiguousLabelClassifier`.

**Problem:** the label encoder spans all six categories in severity order so
codes mean the same thing across cities. But a rare category can fall entirely
inside the test period, leaving `y_train = {0,1,2,4,5}`. XGBoost ≥1.7 rejects
non-contiguous labels outright: *"Invalid classes inferred from unique values
of y"*.

**Solution:** map labels down to a dense `0..k-1` range before fitting and map
predictions back afterwards, so callers and the saved artefact keep working in
the project's canonical label space. `predict_proba` widens back out, giving
unseen classes probability 0 so column *i* always means class *i*.

The wrapper exposes `inner_model` so SHAP can introspect the real estimator.

### 10.5 Selection

Best by `f1_macro` on the held-out test set. For Delhi:

| Model | Macro F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| LogisticRegression | 0.207 | 0.268 | 0.272 | 0.468 |
| RandomForest | 0.262 | 0.296 | 0.287 | 0.498 |
| **XGBoost** | **0.276** | 0.287 | 0.270 | **0.596** |

---

## 11. The Streamlit app, tab by tab

Launch: `streamlit run app/streamlit_app.py`

### Shared shell

- **City selector** at the top switches between all six cities
- **Sidebar** shows the winning model, its macro F1/precision/recall, the test
  period and row counts read from the saved metadata, and a comparison table of
  all three models
- **Graceful degradation:** `city_has_model()` checks **all six** required
  artefacts before taking the "model available" branch, so a half-built city
  shows an explanation instead of crashing inside the loader

**Caching:** `@st.cache_resource` for models (loaded once per process),
`@st.cache_data` for dataframes. Without this, every slider drag would reload a
7 MB CSV and a 6 MB model.

### Tab 1 — Historical View

**Plain:** pick any past date and hour and see what the weather was, what the
pollution actually was, and what the model would have predicted — side by side,
so you can see where it is right and where it is wrong.

**Technical:**
- **"Jump to month"** selector lists only months containing data, so no month
  is reachable only by stepping the calendar back one at a time
- Date/hour picked on the **local** clock; both local and UTC shown
- Nearest-record lookup: `(datetime_local − picked).abs().argsort()[:1]`
- Prediction vs actual, with a success/warning banner
- **SHAP expander** — per-prediction attribution (see §12)
- **Recent trend chart** with EPA threshold lines at AQI 50/100/150/200

### Tab 2 — Forecast View

**Plain:** fetches tomorrow's weather and predicts the air quality risk for the
next few days.

**Technical:**
1. `fetch_weather_forecast(..., forecast_days=N, past_days=8)` — one contiguous
   series spanning 8 real past days plus the horizon
2. Weather rolling/lag features computed over that contiguous block
3. Pollutant features carried forward from the most recent real readings,
   because future pollution is precisely what is being predicted
4. Time features from the local clock
5. **No feature is ever silently filled with zero.** A missing feature raises a
   visible error instead. (The original code backfilled any missing feature with
   `0`, which for `wind_direction_10m` asserted "due north, every hour".)
6. Staleness disclosed **only** when a city's data genuinely stops early

Output: per-hour risk badges, a coloured timeline scatter, and a table — all in
local time, with risk categories ordered by severity.

### Tab 3 — Explainability

**Plain:** shows which weather factors the model relies on most overall.

**Technical:** `global_feature_importance()` — `feature_importances_` for tree
models (mean impurity decrease), mean absolute coefficient across classes for
LogisticRegression. Top 15, horizontal bar chart.

### Tab 4 — City Comparison

**Plain:** puts the cities side by side — mean AQI, median AQI, and how much
time each spends in each risk category.

**Technical:** statistics computed fresh from each city's own
`merged_clean.csv`. The narrative text is **generated from the data** —
it identifies the highest and lowest mean AQI cities and which ones span
multiple risk categories at runtime, so it stays correct as cities are added.
(It was previously hardcoded to a Delhi-vs-Auckland story while the charts
iterated over all six.)

### Tab 5 — Allergy Comparison

**Plain:** a live pollen forecast, plus an honest explanation of how this
project compares with MetService's pollen service.

**Technical:**
- Live species-level pollen from Atmospore, with a visible daily call-budget
  progress bar
- Cache-first: reuses today's cached forecast unless you explicitly refresh
- Status handling for `live` / `cached` / `stale_cache` / `limit_reached` /
  `error`
- A methodology comparison table, and a documented explanation that MetService's
  real-time pollen feed is a licensed commercial API — so rather than shipping a
  fragile scraper, the comparison stays methodology-level and says so

---

## 12. Explainability

### Two complementary views

| | Global | Local |
|---|---|---|
| **Question** | What drives predictions overall? | Why *this* prediction? |
| **Method** | `feature_importances_` / coefficients | SHAP values |
| **Where** | Explainability tab | Expander in Historical View |

### SHAP in practice (`src/explainability.py`)

**Plain:** SHAP assigns each feature a credit or blame score for one specific
prediction — "high humidity pushed this toward Unhealthy, strong wind pushed it
away".

**Technical:** `shap.TreeExplainer` on tree models. Two robustness problems are
handled explicitly:

1. **SHAP's output shape changed across library versions** — older releases
   return a list of per-class arrays, newer ones a single
   `(n_instances, n_features, n_classes)` array, and binary/regression cases a
   2-D array with no class axis. All three are normalised into one flat
   per-feature result.
2. **The XGBoost wrapper** — SHAP inspects the model object directly and knows
   nothing about `ContiguousLabelClassifier`, so `unwrap_model()` hands it the
   inner estimator and `to_dense_class_index()` translates the canonical
   severity code into the dense index the inner model actually uses.

Unsupported model types raise a clear `RuntimeError` that the app catches and
displays, rather than crashing the page.

---

## 13. Validation and testing

### 13.1 The AQI cross-check (`src/validate_aqi_crosscheck.py`)

**Plain:** we check our own AQI maths against a completely independent source.

**Technical:** compares our computed AQI against Open-Meteo's `us_aqi` over a
14-day window, reporting **two** comparisons because they measure different
things:

- **instantaneous** — our AQI from each hour's raw concentration. This is what
  the pipeline labels on, and it is *expected* to disagree: EPA's PM
  breakpoints are defined over 24-hour averages, and Open-Meteo applies that
  averaging (NowCast) while we apply the table to a single hourly reading.
- **averaged_24h** — our AQI from a trailing 24-hour mean. Isolates the
  breakpoint arithmetic from the averaging choice.

#### The verdict rests on an invariant, not a tuned threshold

`us_aqi` is a maximum over roughly six pollutant sub-indices (PM2.5, PM10, NO2,
O3, SO2, CO). We take the maximum over the three we collect. **A maximum over a
subset cannot exceed a maximum over the superset**, so our AQI must sit at or
below theirs — and a breakpoint error would break that immediately, in one
direction.

MAE and correlation are reported for context but are deliberately *not* the
gate, because they swing with the season rather than with our arithmetic:

| Window | MAE | Correlation |
|---|---|---|
| January | 1.4 | 0.998 |
| July | 31 | 0.978 |
| September | 10 | 0.875 |

The invariant held in all three. (This also confirmed the 2024 breakpoint table
is the right default: January matches `us_aqi` to 1.4 points under 2024 versus
16.5 under the pre-2024 table.)

### 13.2 The test suite — 69 tests

```bash
python -m pytest
```

| File | Covers |
|---|---|
| `test_labeling.py` | Breakpoint gaps across all four tables, monotonicity, category banding, the 2024-vs-2012 difference, severity-ordered encoding |
| `test_feature_engineering.py` | Leakage exclusion, true-hour windows, gap non-bridging, DST, hemisphere seasons, dummy-column completeness |
| `test_pipeline_parts.py` | OpenAQ pagination, station selection, coverage assessment, data-derived split, the XGBoost wrapper, CSV datetime parsing |

Every test pins down something that was genuinely wrong at some point. For
example:

```python
def test_no_concentration_falls_into_a_breakpoint_gap(...):
    """Every non-negative concentration must produce a usable sub-index."""
    offenders = [v for v in np.arange(0, hi, step)
                 if np.isnan(_sub_index(v, breakpoints, decimals))]
    assert offenders == []
```

---

## 14. Results

### Per-city summary

| City | Source | Hours | Mean AQI | Classes | Best model | Macro F1 | Accuracy |
|---|---|---|---|---|---|---|---|
| Delhi | ground | 8,663 | 167.4 | 6 | XGBoost | 0.276 | 0.596 |
| Auckland | ground | 8,661 | 10.6 | 3 | XGBoost | 0.523 | 0.894 |
| Wellington | model | 8,664 | 27.0 | 2 | LogisticRegression | 0.666 | 0.880 |
| Christchurch | ground | 8,663 | 32.9 | 5 | XGBoost | 0.350 | 0.775 |
| Hamilton | model | 8,664 | 27.5 | 2 | LogisticRegression | 0.584 | 0.624 |
| Dunedin | ground | 8,661 | 14.4 | 4 | LogisticRegression | 0.385 | 0.769 |

All six at 100% timeline coverage, all with a data-derived split at 2026-06-28.

### How to read these numbers honestly

- **Delhi's 0.276 is the hardest and most meaningful score.** It has all six
  risk classes across a genuinely variable year. The NZ cities score higher
  largely because they are 78–97% "Good" — a two-class problem is easier.
- **Ground-sensor and model-estimated cities are not strictly comparable.**
  Wellington and Hamilton use Open-Meteo's model output rather than physical
  sensors.
- **Modest F1 is honest, not a failure.** Weather is a real but partial
  predictor of pollution. Traffic, industry and construction drive a large share
  of Delhi's air quality and none of them are collected here.

### The Delhi contrast

Delhi's mean AQI of 167.4 against Auckland's 10.6 — a 15.7× gap — is the
core motivation for a multi-city design: one city with enough variance to make a
real prediction problem, and several clean enough that the problem nearly
disappears.

---

## 15. Engineering decisions and bugs fixed

This section is the most valuable part for a presentation — it shows
investigative work, not just assembly.

### 15.1 The pagination bug that silently discarded 40% of history

**Symptom:** Delhi had 60.4% timeline coverage with contiguous multi-week holes.

**Cause:** OpenAQ v3 returns `meta.found` as the **string** `">1000"` once a
result set exceeds one page. The break condition was:

```python
found = meta.get("found", 0)
if page * 1000 >= (found if isinstance(found, int) else 0) or len(results) < 1000:
    break
```

`isinstance(">1000", int)` is `False` → `found` became `0` → `page * 1000 >= 0`
is always true → **the loop broke after page 1**, keeping only the first 1,000
hours of every 2,160-hour chunk.

**Fix:** break only on a short page. Never consult `meta.found`.

**Impact:** coverage 60.4% → 96.7%; the lost blocks included the start of
stubble-burning season and peak winter smog — exactly where the Very Unhealthy
and Hazardous hours live.

### 15.2 Two different clocks joined as one

**Cause:** Open-Meteo was asked for `timezone=auto` (city-local), while OpenAQ
reports UTC. `merge_clean.py` inner-joined them on `datetime` as if they shared
a clock — offsetting every label from its features by 5.5 h for Delhi and
12–13 h for the NZ cities (non-constant, because NZ observes DST).

**Evidence:** Delhi weather peaked at hour 14 and bottomed at hour 5 — that is
IST, not UTC. Shifting the joined series to correct for it strengthened the
temperature↔PM2.5 anticorrelation from −0.471 to −0.563.

**Fix:** fetch everything in UTC; derive local-clock features separately through
each city's IANA timezone. After the fix, Delhi's weather peaks at hour 9 UTC
(= 14:30 IST). ✓

### 15.3 Target leakage through rolling features

Covered in §8.3. Verified against the stored data:

```
row  500: stored=47.1438  mean(t−23..t)=47.1438  mean(t−24..t−1)=46.8729  → INCLUSIVE
```

The stored value matched the window *including* the current row exactly.

### 15.4 A train/test split that moved with the wall clock

`SPLIT_DATE = END_DATE − 90 days` where `END_DATE = date.today() − 7`. Nothing
referred to the dataset, so as the data aged the "last 3 months" test window
silently shrank — the saved Delhi model had been evaluated on **461 rows
(≈19 days)** while the ReadMe claimed three months — and would eventually become
empty.

**Fix:** derive from `df.datetime.max()` and persist into
`model_comparison.json`. Delhi's test set went from 461 rows to **2,089 rows
(89 days)**.

### 15.5 Station slots spent on sensors dead since 2016

Delhi has **96** OpenAQ stations within 25 km; only 6 slots are available.
Sorting by distance alone gave three of them to stations last heard from in
2016, 2018 and 2018.

**Fix:** skip stations whose `datetimeLast` predates `START_DATE`, rank
confirmed-live by distance, and print each chosen station with its last-report
date so this cannot silently rot again.

**Impact:** 30 dead stations skipped, 6 live ones selected; coverage 96.7% →
**100.0%**, macro F1 0.225 → **0.276**, accuracy 0.503 → **0.596**.

### 15.6 "Station exists" ≠ "usable data"

Hamilton has two stations 0.7 km and 2.7 km away. Both stopped on 2025-12-03.
The fallback only triggered when **zero** stations were found, so Hamilton kept
2 months of a 12-month window — a Historical tab stuck in 2025 and a forecast
resting on a 10-month-old pollution baseline.

**Fix:** gate the fallback on measured coverage and staleness rather than
existence. Hamilton now reports `REJECTED: covers only 18%` and falls back,
giving it the full 8,622 rows.

### 15.7 A retracted finding — Auckland

An earlier version of this project reported that Auckland's entire dataset fell
into a single risk category (Good, 100%, mean AQI 7.9) and presented it as a
real result: the clean-air city where the prediction problem disappears.

**That was an artifact of the pagination bug**, not a finding. Auckland was
being trained on 4,083 of the 8,661 hours actually available, and the missing
53% contained the Moderate hours. With complete data it is 96.2% Good / 3.8%
Moderate, mean AQI 10.6 — two classes, so it trains like any other city.

Confirmed this was *not* caused by the 2024 breakpoint change: the pre-2024
table yields identical class counts on the same data.

**What survived:** the clean-air contrast, decisively — Delhi 167.4 against
Auckland 10.6, a **15.7× gap**.
**What was overstated:** the size of it. The original 153.7 vs 7.9 worked out
at 19.5×, because Auckland's mean was computed on the truncated 47% of its data.
With the full year its mean rises to 10.6, so the real ratio is smaller.
**What did not survive at all:** "single class, so there is nothing to learn."

This is why `merge_clean.py` now prints timeline coverage on every run and warns
below 85%. Auckland's would have read 47%.

### 15.8 The staleness warning that fired every single time

A check compared the last reading against `now` with a 48-hour threshold. But
`END_DATE = date.today() − 7`, so the freshest *possible* reading is always
about a week old. The threshold sat below the pipeline's own structural lag, so
it fired on every forecast for every city — including moments after a clean
run — and advised re-running the pipeline, which could not have changed
anything.

**Fix:** measure against `END_DATE` (what the pipeline can collect), not `now`.
Silent for every normally-collecting city; disclosed only when data genuinely
stops early.

### 15.9 Other fixes

| Issue | Resolution |
|---|---|
| Forecast fed `wind_direction_10m = 0` | Use the full configured variable list; never backfill a missing feature with zero |
| Forecast rolling windows bridged a 2-month gap | Fetch with `past_days=8` for one contiguous series |
| IQR clipping leaked and removed extreme classes | Replaced with physical sanity ceilings |
| Macro F1 deflated by an absent class | Report both present-classes and all-labels variants |
| Northern-hemisphere seasons for NZ | Gate on latitude |
| `use_container_width` deprecated | Switched to `width="stretch"`, pinned Streamlit ≥1.49 |
| `has_model` under-checked artefacts | Require all six |
| No tests at all | 69 regression tests |

---

## 16. Deployment and operations

### Pre-deployment status

| Check | Status |
|---|---|
| Secrets committed | ✅ None — `.env` is gitignored and untracked |
| Largest tracked file | ✅ 7.6 MB (GitHub limit 100 MB) |
| App runs without the pipeline | ✅ Reads committed CSVs/models |
| Tests passing | ✅ 69/69 |
| Lint | ✅ Clean |
| App exceptions | ✅ Zero across all six cities |

### Running it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# keys (free): OpenAQ for the pipeline, Atmospore for the Allergy tab
cp .env.example .env     # then fill in

python -m pytest                      # verify
python run_pipeline.py --all          # collect + train all six cities (45-90 min)
streamlit run app/streamlit_app.py    # launch
```

### Operational notes

- **The app does not need the OpenAQ key.** Only the data pipeline calls
  OpenAQ. A deployed app serving committed data runs without it.
- **The Allergy tab needs the Atmospore key**, and degrades to a warning
  without one.
- **On Streamlit Community Cloud**, set keys via the Secrets UI rather than a
  `.env` file.
- **The data window slides.** `START_DATE` derives from `date.today()`, so the
  earliest available date advances daily. Re-run periodically, or pin
  `START_DATE`/`END_DATE` to fixed dates for a reproducible range.
- **Repo weight:** ~65 MB of committed data and models. Fine for GitHub, but it
  makes clones heavy. Moving to Git LFS or regenerating data on deploy are both
  reasonable alternatives.

---

## 17. Limitations and honest caveats

1. **Weather alone is a partial predictor.** Traffic, industry and construction
   are not collected. This caps achievable performance and is the main reason
   Delhi's F1 is modest.
2. **Mixed data provenance.** Four cities use ground sensors, two use
   model-estimated data. Disclosed per row via the `source` column, but it means
   cross-city scores are not strictly comparable.
3. **Not every city measures every pollutant.** A city with only a PM2.5 sensor
   gets a PM2.5-only AQI, which slightly understates it — so a cross-city AQI
   gap is a floor on the real difference, not an exact ratio.
4. **Hourly readings against 24-hour breakpoints.** EPA's PM breakpoints are
   defined over 24-hour averages; applying them to hourly readings is standard
   practice for near-real-time estimation but is an approximation, and it is why
   the instantaneous cross-check disagrees with `us_aqi`.
5. **Rare classes stay hard.** Good, Very Unhealthy and Hazardous have very few
   test rows in Delhi, so their per-class F1 is near zero and the macro average
   suffers.
6. **The forecast carries pollution history forward.** Future pollution is what
   is being predicted, so its rolling/lag features assume the recent past
   persists.
7. **12-month window.** Only one instance of each season, so the model cannot
   distinguish a seasonal pattern from a one-off year.

### Natural extensions

- Traffic volume or satellite NO2 as additional features
- Multi-year collection for genuine seasonal validation
- Hyperparameter search (nothing is currently tuned)
- Predicting AQI as a regression target, then banding
- Probability calibration, so "70% Unhealthy" means something

---

## 18. File-by-file map

| File | Lines | Purpose |
|---|---|---|
| `config.py` | 211 | City registry with timezones, API endpoints, thresholds, per-city paths |
| `run_pipeline.py` | 122 | Per-city pipeline driver (`--city` / `--all` / `--skip-fetch`) |
| `src/utils.py` | 139 | HTTP retry/backoff, haversine, date chunking, split derivation, UTC→local, CSV reader |
| `src/fetch_weather.py` | 102 | Historical + forecast weather (UTC) |
| `src/fetch_air_quality.py` | 362 | OpenAQ discovery, station filtering, pagination, coverage gate |
| `src/fetch_air_quality_openmeteo.py` | 95 | Open-Meteo Air Quality fallback source |
| `src/fetch_pollen.py` | 316 | Atmospore pollen with call budget + cache |
| `src/labeling.py` | 263 | EPA breakpoints, AQI, risk categories, ordered encoder |
| `src/merge_clean.py` | 161 | UTC join, cleaning, labelling, coverage report |
| `src/feature_engineering.py` | 218 | Time/rolling/lag features on an hourly grid |
| `src/train_models.py` | 283 | Split, train, compare, select, persist |
| `src/model_wrappers.py` | 69 | XGBoost label-space adapter |
| `src/explainability.py` | 160 | Global importance + SHAP normalisation |
| `src/validate_aqi_crosscheck.py` | 220 | Independent AQI validation |
| `app/streamlit_app.py` | 876 | The entire web application |
| `tests/` | 623 | 69 regression tests |

### Data artefacts per city

```
data/raw/<city>/weather_raw.csv            ~8,664 rows
data/raw/<city>/air_quality_raw.csv        ~8,664 rows
data/processed/<city>/merged_clean.csv     ~8,663 rows, labelled
data/processed/<city>/features.csv         ~8,168 rows × 77 columns
data/processed/<city>/aqi_crosscheck_report.json
models/<city>/best_model.joblib
models/<city>/scaler.joblib
models/<city>/label_encoder.joblib
models/<city>/feature_list.json
models/<city>/model_comparison.json
```

---

## 19. Glossary

| Term | Plain meaning |
|---|---|
| **AQI** | Air Quality Index — pollution on a single 0–500 scale |
| **PM2.5 / PM10** | Airborne particles under 2.5 / 10 micrometres across |
| **NO2** | Nitrogen dioxide — largely from vehicle exhaust |
| **Breakpoint** | A boundary in the table mapping concentration to AQI |
| **Sub-index** | The AQI score for one pollutant on its own |
| **Feature** | One input number the model learns from |
| **Label / target** | The answer being predicted — here, the risk category |
| **Target leakage** | A feature secretly containing the answer; inflates scores and fails in production |
| **Rolling feature** | A statistic over a trailing time window |
| **Lag feature** | The value from exactly N hours ago |
| **Chronological split** | Train on older data, test on newer |
| **Macro F1** | Per-class F1 averaged equally, so rare classes count |
| **Class imbalance** | Some categories far more common than others |
| **SHAP** | A method attributing one prediction to its features |
| **UTC** | The global reference clock |
| **DST** | Daylight Saving Time — why offsets are not constant |
| **IANA timezone** | A named zone like `Pacific/Auckland` that encodes DST rules |

---

## 20. Suggested presentation outline

A 15–20 slide structure built from this report.

| # | Slide | Source |
|---|---|---|
| 1 | Title — Air Quality Risk Predictor | — |
| 2 | The problem: can weather predict air quality? | §1, §2 |
| 3 | Why it's harder than it looks (7 challenges) | §2 table |
| 4 | Architecture diagram | §4 |
| 5 | Data sources — 5 APIs | §5 |
| 6 | From concentration to risk: the AQI maths | §7 |
| 7 | Feature engineering — 64 features | §8 table |
| 8 | **Preventing target leakage** | §8.3 |
| 9 | Chronological split, and why not random | §9.1 |
| 10 | Three models compared | §10, §10.5 table |
| 11 | App demo — Historical View | §11 |
| 12 | App demo — Forecast View | §11 |
| 13 | App demo — Explainability + SHAP | §11, §12 |
| 14 | App demo — City Comparison | §11 |
| 15 | Results across six cities | §14 table |
| 16 | **War story: the bug that hid 40% of the data** | §15.1 |
| 17 | **War story: the dead-sensor problem** | §15.5, §15.6 |
| 18 | **Retracting a finding — scientific honesty** | §15.7 |
| 19 | Validation: the invariant-based cross-check | §13.1 |
| 20 | Limitations and future work | §17 |

### Slides that will stand out

Most student projects present a pipeline and an accuracy number. The
differentiators here are:

- **Slide 8 (leakage)** — shows you understand *why* a high score can be fake
- **Slide 16 (pagination bug)** — real debugging, with before/after evidence
- **Slide 18 (retraction)** — a previously-reported finding withdrawn after
  investigation. Very few projects demonstrate this kind of honesty, and it is
  genuinely the strongest thing here.
- **Slide 19 (invariant validation)** — validating against a mathematical
  property rather than a tuned threshold is a mature testing idea

### The one-sentence pitch

> A supervised ML system that predicts air quality risk from weather across six
> cities — where most of the engineering went into making sure the model was
> learning from honest data rather than from its own answer.

---

*Report generated October 2026. Figures reflect the state after the full
six-city collection run; re-run `python run_pipeline.py --all` and read
`models/<city>/model_comparison.json` for current numbers.*

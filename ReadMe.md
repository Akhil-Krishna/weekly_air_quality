# Air Quality Risk Predictor

An end-to-end supervised machine learning system that predicts air quality
risk levels from weather data, with a Streamlit app for interactive
prediction, model explainability, and cross-city comparison.

Built for Delhi (high pollution variance) plus five New Zealand cities
(Auckland, Wellington, Christchurch, Hamilton, Dunedin) as clean-air
baselines. All six train a model.

---

## What this project actually does

1. Collects real hourly weather (Open-Meteo) and real air pollution readings
   (OpenAQ ground stations, with an automatic fallback to Open-Meteo's
   model-based Air Quality API for cities with no nearby station). Everything
   is fetched and stored in **UTC**, which is what makes the two sources
   joinable.
2. Converts raw pollutant concentrations into an AQI value and a risk
   category (Good / Moderate / Unhealthy for Sensitive Groups / Unhealthy /
   Very Unhealthy / Hazardous), using the **US EPA 2024** breakpoint tables.
3. Cleans and merges the two datasets, engineers time-based and
   rolling/lag features, and trains three models (Logistic Regression,
   Random Forest, XGBoost) on a **chronological** train/test split so the
   model is evaluated the way it would actually be used -- predicting the
   future from the past.
4. Serves everything through a Streamlit app: historical lookup, live
   forecast prediction, model explainability, a cross-city comparison, and a
   methodology comparison against MetService's pollen forecasting.

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
2. Copy `.env.example` to `.env` and put the key in it, or set
   `OPENAQ_API_KEY` as an environment variable.

### 3. Run the pipeline

```bash
python run_pipeline.py --city Delhi     # one city
python run_pipeline.py --all            # every city in config.CITIES
python run_pipeline.py                  # config.DEFAULT_CITY
```

This fetches weather, fetches air quality (OpenAQ first, falling back to
Open-Meteo Air Quality automatically if no station is nearby), merges and
cleans the data, engineers features, trains and compares all three models,
and saves the best one. Every city gets its own folder automatically
(`data/raw/<city>/`, `data/processed/<city>/`, `models/<city>/`) so running
this for a second city never overwrites the first. Each city runs in its own
subprocess, so one failure doesn't take the rest down.

`--skip-fetch` re-derives the merge, features and models from the raw CSVs
already on disk, without re-hitting the APIs. Useful while iterating -- but
only valid on raw data collected *after* the UTC change described below.

To add a city, add it to the `CITIES` dict in `config.py` with its
coordinates **and its IANA timezone**:

```python
CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "tz": "Asia/Kolkata"},
    "Auckland": {"lat": -36.8485, "lon": 174.7633, "tz": "Pacific/Auckland"},
}
```

The timezone isn't decoration: it's what the hour-of-day and season features
are derived from, and it's what makes DST correct for the NZ cities.

### 4. Run the tests

```bash
python -m pytest
```

58 tests covering the AQI breakpoint math, the label space, the feature
builder's leakage and windowing guarantees, the OpenAQ paginator, and the
data-derived split. These are regression tests for real bugs, not padding --
each one pins down something that was previously wrong.

### 5. Launch the app

```bash
streamlit run app/streamlit_app.py
```

---

## The app

Five tabs, with a city selector at the top that switches between any city
that's been through the pipeline:

| Tab | What it does |
|---|---|
| **Historical View** | Pick a past date/hour (in the city's local time), see the actual weather, actual pollution reading, and the model's prediction next to what really happened. A "Jump to month" selector lists only months that contain data, so no month is reachable only by stepping the calendar back one at a time. |
| **Forecast View** | Pulls live upcoming weather from Open-Meteo and predicts risk up to 7 days ahead -- a genuine decision-support tool, not just a dashboard of the past. |
| **Explainability** | Shows which features actually drive the model's predictions (feature importance on the winning model). Per-prediction SHAP breakdowns -- why the model predicted a specific category for one specific date/hour -- are available as an expander inside the Historical View tab. |
| **City Comparison** | Real computed statistics from each city's own data -- not illustrative numbers. The narrative is generated from the data, so it stays correct as cities are added. |
| **Allergy Comparison** | Live species-level pollen forecasts (Atmospore API, with a daily call budget and cache fallback), plus how this project's approach compares to MetService's pollen/allergy forecasting. |

A city with data but no trained model doesn't break the app -- each tab
explains why, using that city's real numbers, instead of crashing or hiding
the city entirely. No city is currently in that state (Auckland used to be;
see below), but the path is still exercised whenever a newly added city has
too little data to train on.

---

## Auckland: a retracted finding

An earlier version of this ReadMe reported that Auckland's entire dataset fell
into a single risk category (**Good**, 100% of hours, mean AQI 7.9), and
presented that as a real result -- the clean-air city where the prediction
problem disappears entirely.

**That was an artifact of a data collection bug, not a finding.** The OpenAQ
paginator was stopping after the first page of each 90-day request, so
Auckland was being trained on 4083 hours out of the 8661 that were actually
available. The missing 53% contained the Moderate hours.

With complete data, Auckland is:

| Risk category | Share of hours |
|---|---|
| Good | 96.2% |
| Moderate | 3.8% |
| Unhealthy (Sensitive Groups) | 1 hour |

Mean AQI 10.6. Two classes is enough to train on, so Auckland now has a model
like every other city.

Worth being precise about what survived and what didn't:

- **The clean-air contrast is intact.** Delhi's mean AQI is 167.4 against
  Auckland's 10.6 -- a 15.7x gap, on 8663 and 8661 hours respectively. Note the
  original 153.7 vs 7.9 figures worked out at 19.5x, so that number was
  overstated: Auckland's mean was computed on the truncated 47% of its data, and
  rises to 10.6 once the full year is present. The contrast survives decisively;
  its exact size was flattered by the same bug.
- **The "single class, so nothing to learn" claim is withdrawn.** It described
  a truncated dataset. This was not caused by the 2024 breakpoint change
  either -- the pre-2024 table yields the same class counts on the same data,
  so the only cause was the missing hours.
- **The single-class guard in `train_models.py` is still correct code**, and
  still runs. It detects the condition upfront and skips training with an
  explanation rather than crashing (which is scikit-learn's default on
  single-class input) or fitting a model that trivially always predicts one
  thing. It just no longer fires for any city in this project.

The general lesson is the one now wired into `merge_clean.py`: it prints
timeline coverage on every run and warns below 85%. Auckland's coverage would
have read 47%, which is the number that should have prompted a look at the
collection step before any conclusion was drawn from the class distribution.

---

## Results

Run `python run_pipeline.py --all` and read the numbers out of
`models/<city>/model_comparison.json`. That file now records the split date,
the train/test row counts and date ranges, the class order, and the
breakpoint-table version alongside the metrics -- so a saved model always
carries the context needed to interpret its own scores.

Air quality provenance differs by city, and is recorded per row in the
`source` column rather than assumed:

| City | Pollution source | Why |
|---|---|---|
| Delhi | `openaq_ground` | 64 live stations within 25 km; the 6 nearest live ones are used |
| Auckland | `openaq_ground` | live station coverage |
| Christchurch | `openaq_ground` | live station coverage |
| Dunedin | `openaq_ground` | live station coverage |
| Wellington | `openmeteo_model` | OpenAQ returns no stations within 25 km |
| Hamilton | `openmeteo_model` | both nearby stations stopped reporting on 2025-12-03 |

Ground-sensor and model-estimated readings are different in kind, so the two
groups are not strictly comparable with each other -- treat the per-city
numbers as within-source results.

How to read the two F1 figures it reports:

- **`f1_macro`** averages only over the risk categories actually present in
  the test set. This is the headline number.
- **`f1_macro_all_labels`** averages over all six categories. A category with
  zero test rows contributes a guaranteed 0.0 to that mean, so this figure is
  depressed by a property of the split rather than by the model.

F1 scores on this problem are modest, and that's honest rather than a bug to
hide: weather alone is a real but partial predictor of pollution -- local
traffic, industrial activity, and construction (none of which this project
collects) also drive Delhi's air quality.

**AQI calculation cross-check:** `src/validate_aqi_crosscheck.py` validates
our own EPA-breakpoint AQI math against Open-Meteo's independently-computed
`us_aqi`, and reports **two** comparisons, because they measure different
things:

- *instantaneous* -- our AQI from each hour's raw concentration against their
  `us_aqi`. This is what the pipeline labels on, and it is *expected* to
  disagree: EPA's PM breakpoints are defined over 24-hour averages, and
  Open-Meteo applies that averaging (NowCast) while `labeling.py` applies the
  table to a single hourly reading.
- *averaged_24h* -- our AQI from a trailing 24h mean concentration against the
  same `us_aqi`. This isolates the breakpoint arithmetic from the averaging
  choice, so it's the comparison that actually tests `labeling.py`, and it's
  the one the pass/fail verdict is applied to.

The verdict rests on an **invariant**, not a tuned error threshold.
Open-Meteo's `us_aqi` is a max over roughly six pollutant sub-indices (PM2.5,
PM10, NO2, O3, SO2, CO); `labeling.py` takes the max over the three pollutants
this project collects. A max over a subset cannot exceed a max over the
superset, so our AQI must sit at or below theirs -- and a breakpoint-table
error would break that immediately, in one direction.

MAE and correlation are reported for context but deliberately aren't the gate.
Measured over three 14-day Delhi windows they ranged from MAE 1.4 / corr 0.998
(January) to MAE 31 / corr 0.978 (July) to MAE 10 / corr 0.875 (September) --
moving with which pollutant happened to drive Open-Meteo's max, not with
anything about our arithmetic. The invariant held in every window.

The report writes an explicit `verdict` field (`consistent` or `needs_review`)
so the script's own assessment can't be read backwards, as it was previously:
the old version printed "difference is fairly large -- worth double-checking
labeling.py breakpoints" while the ReadMe cited the same numbers as evidence
the labeling logic was correct.

Incidentally, this also confirms the 2024 table is the right default: on the
January window our AQI matches `us_aqi` to within 1.4 points under the 2024
table versus 16.5 under the pre-2024 one.

---

## Project structure

```
weekly_air_quality/
├── config.py                    # city registry, API endpoints, file paths
├── run_pipeline.py              # per-city pipeline driver (--city / --all)
├── pytest.ini
├── requirements.txt
├── .env.example
├── src/
│   ├── fetch_weather.py
│   ├── fetch_air_quality.py
│   ├── fetch_air_quality_openmeteo.py
│   ├── fetch_pollen.py
│   ├── labeling.py                  # EPA breakpoints + severity-ordered labels
│   ├── merge_clean.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── model_wrappers.py            # XGBoost label-space adapter
│   ├── explainability.py            # feature importance + SHAP
│   ├── validate_aqi_crosscheck.py   # validates labeling.py against Open-Meteo
│   └── utils.py
├── app/
│   └── streamlit_app.py
├── tests/                       # regression tests for the bugs below
├── data/{raw,processed}/<city>/ # per-city data, kept separate automatically
├── models/<city>/               # per-city trained model + metrics
└── demo_week4.py / demo_week5.py / demo_week6.py   # weekly progress demos
```

---

## Design notes and honest limitations

- **One clock: UTC.** Weather and pollution are fetched, stored and joined in
  UTC. Hour-of-day, day-of-week and season are derived from the city's local
  time (via its IANA timezone, so DST is handled) and stored alongside as
  `datetime_local`. These have to be separated: the join key needs a single
  global clock, while "rush hour" and "winter" are local concepts.
- **Chronological split, not random.** The model is tested on the most
  recent months only, held out entirely from training -- this mirrors how
  it would actually be used (predicting the future from the past) and
  avoids the leakage a random shuffle-split would introduce. The split point
  is derived from the data's own date range and persisted into
  `model_comparison.json`, so the app reports the split the saved model was
  genuinely evaluated on.
- **No target leakage through rolling features.** The risk label is computed
  from the current hour's PM2.5/PM10/NO2, so rolling averages of those
  pollutants are shifted back one hour and exclude the current observation.
  Excluding the raw pollutant columns from the feature list is not sufficient
  on its own -- a 24-hour mean that includes the present moment still carries
  the answer.
- **Rolling/lag windows are hours, not rows.** Features are computed on a
  gap-filled hourly grid, so `pm25_roll_24h_mean` really is 24 hours and
  `pm25_lag_24h` really is the reading 24 hours earlier. Rows that fall after
  a gap longer than the window are dropped rather than given a window
  silently spanning weeks.
- **Macro F1 over accuracy.** Risk categories are heavily imbalanced (Delhi
  is mostly Unhealthy/Moderate; the NZ cities are 78-97% Good) -- accuracy
  would hide poor performance on rarer classes. Auckland's model scores
  0.894 accuracy against 0.523 macro F1, which is that gap in two numbers. `class_weight="balanced"`
  is used where supported. Accuracy is reported too, for context.
- **Outlier handling keeps the tail.** Only physically impossible pollutant
  readings are rejected (negatives, and values above the sanity ceilings in
  `config.POLLUTANT_SANITY_CEILINGS`). IQR winsorizing was removed: it fit a
  cleaning parameter across train and test together, and for a heavy-tailed
  quantity like Delhi PM2.5 it clipped away exactly the extreme readings the
  Very Unhealthy and Hazardous classes are made of.
- **Hybrid air quality source, disclosed per row.** Every row carries a
  `source` column (`openaq_ground` or `openmeteo_model`) recording which
  data source was actually used -- ground-truth sensor data and
  model-estimated data are genuinely different in kind, and this is never
  hidden.
- **Station slots go to stations that are still reporting.** Delhi has 96
  OpenAQ stations within 25 km and `OPENAQ_MAX_LOCATIONS` allows 6. Choosing
  them by distance alone gave three slots to stations whose last readings were
  in 2016, 2018 and 2018 -- half the request budget returned nothing, while
  ~90 live stations slightly further out were never queried. Stations that
  stopped reporting before `START_DATE` are now skipped, and the chosen
  stations are printed with their last-report date on every run.
- **The ground-data fallback is based on coverage, not existence.** Asking
  "is there a station nearby" is the wrong question. Hamilton has two stations
  0.7 km and 2.7 km away that both stopped on 2025-12-03, so the existence
  check passed while the city received 2 months of a 12-month window -- a
  Historical tab stuck in 2025 and a forecast resting on a 10-month-old
  pollution baseline. OpenAQ data is now accepted only if it covers at least
  `OPENAQ_MIN_WINDOW_COVERAGE` of the requested hours **and** reaches within
  `OPENAQ_MAX_STALENESS_DAYS` of the end of the window; otherwise the pipeline
  falls back to Open-Meteo and says so. The two conditions catch different
  failures: coverage catches a station that reports sporadically, staleness
  catches one that reported well and then died.
- **The collection window slides.** `START_DATE` is derived from
  `END_DATE = date.today() - 7`, which is itself ~360 days wide, so the
  earliest available date advances every day the pipeline is not re-run. The
  Historical tab's date picker reflects exactly what was collected, which is
  why dates before roughly a year ago are greyed out rather than empty. Note
  the edge this creates: the earliest available date is partway through a
  month (1 Oct 2025 at the time of writing), so opening the calendar's year
  dropdown and selecting the earliest year lands on a month that is entirely
  out of range and renders fully greyed -- which reads as "that whole year is
  unavailable" when it is not. The "Jump to month" selector exists to make
  that impossible to hit.
  Widen it with `HISTORY_MONTHS`, or pin `START_DATE`/`END_DATE` to fixed
  dates if you want a reproducible range.
- **Not every city measures the same pollutants.** Delhi's AQI is the max of
  PM2.5/PM10/NO2 sub-indices; a city with only a PM2.5 sensor gets a PM2.5-only
  AQI. That slightly understates AQI for the sparser cities, so a cross-city
  AQI gap is a floor on the real difference, not an exact ratio.
- **Forecast view assumptions.** Future weather comes from a real forecast,
  fetched with `past_days` so the 24h/7d rolling weather features are computed
  over one contiguous series rather than spliced across a stale join. Future
  pollution is exactly what's being predicted, so the pollutant rolling/lag
  features carry forward the most recent known real values. No feature is ever
  silently filled with zero: a missing feature raises in the UI instead.
- **Data staleness is measured against `END_DATE`, not against now.**
  `END_DATE` is `date.today() - 7` because the archive APIs lag real time, so
  the freshest possible reading is always about a week old. A check that
  compared the last reading to the current time with a 48-hour threshold
  therefore fired on every forecast for every city -- including moments after
  a clean run -- and advised re-running the pipeline, which could not have
  changed anything. The forecast tab now compares against `END_DATE` and stays
  silent for any city collecting normally, disclosing the gap only when a
  city's data genuinely stops early.
- **PM2.5 breakpoint vintage is explicit.** `config.PM25_BREAKPOINT_VERSION`
  selects EPA's 2024 table (default) or the pre-2024 one. The 2024 revision
  moved the Good/Moderate boundary from 12.0 to 9.0 µg/m³ and restructured the
  upper bands, so which table is in use materially changes the labels.
- **Timeline coverage is reported on every run.** `merge_clean.py` prints what
  fraction of the nominal hourly timeline survived and warns below 85%. Low
  coverage is the signal that collection is dropping data, and it was: the
  OpenAQ paginator used to stop after one page per 90-day chunk.
- **Allergy comparison is methodology-level for MetService.** Live pollen comes
  from Atmospore. MetService's real-time pollen feed is a licensed commercial
  API, not a public one, so that half of the comparison stays honest about
  what it is rather than shipping a fragile scraper.

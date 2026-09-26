"""
Pollen forecasting via the Atmospore API, with a locally-enforced daily call
budget and on-disk caching -- so a single free-tier API key (100 calls/day,
shared across EVERY city this app supports, not per-city) can safely back a
multi-city Streamlit app that reruns its whole script on every widget click.

Strategy, in priority order, every time pollen data is requested for a city:
  1. If we already fetched that city's forecast TODAY (UTC), reuse it --
     Atmospore's own data only updates once a day, so re-fetching within
     the same day buys nothing and just burns quota for no benefit.
  2. Otherwise, check the shared call ledger (one counter for the whole
     API key, reset at UTC midnight). If there's budget left, make one
     live call, cache the result, and increment the ledger.
  3. If the budget is exhausted (or the live call itself gets a 429), fall
     back to the most recent cache for that city, however old it is,
     clearly labeled as stale so the UI can say so honestly.
  4. If there is no cache at all yet for that city, report the limit
     plainly instead of crashing -- the caller decides how to degrade
     (e.g. fall back to static seasonal info).

get_pollen_forecast() never raises. It always returns a status dict the
caller can render directly.
"""

import os
import sys
import json
import datetime as dt
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

POLLEN_RISK_COLORS = {
    "low": "#00A651",
    "moderate": "#FFD400",
    "high": "#FF8C00",
    "very high": "#E3312C",
}


class _RateLimited(Exception):
    """Raised internally when Atmospore itself returns 429."""
    pass


def _today_str():
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def _cache_path(city_slug):
    return os.path.join(config.ATMOSPORE_CACHE_DIR, f"{city_slug}.json")


def _load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _age_hours(iso_ts):
    if not iso_ts:
        return None
    then = dt.datetime.fromisoformat(iso_ts)
    return (dt.datetime.utcnow() - then).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Shared daily call ledger (one counter for the whole API key)
# ---------------------------------------------------------------------------
_LEDGER_PATH = os.path.join(config.ATMOSPORE_CACHE_DIR, "_call_ledger.json")


def _load_ledger():
    ledger = _load_json(_LEDGER_PATH)
    today = _today_str()
    if not ledger or ledger.get("date") != today:
        ledger = {"date": today, "count": 0}
        _save_json(_LEDGER_PATH, ledger)
    return ledger


def _increment_ledger():
    ledger = _load_ledger()
    ledger["count"] += 1
    _save_json(_LEDGER_PATH, ledger)
    return ledger["count"]


def _mark_ledger_exhausted():
    """Called on a genuine 429 from the server -- trust the server over our
    own count, in case another process (or a restart that lost local state)
    already spent calls today."""
    ledger = _load_ledger()
    ledger["count"] = max(ledger["count"], config.ATMOSPORE_DAILY_CALL_BUDGET)
    _save_json(_LEDGER_PATH, ledger)


def get_call_budget_status():
    """Cheap, side-effect-free peek at today's usage -- safe to call on
    every Streamlit rerun to render a usage indicator without touching the
    network."""
    ledger = _load_ledger()
    used = ledger["count"]
    budget = config.ATMOSPORE_DAILY_CALL_BUDGET
    return {
        "calls_used_today": used,
        "calls_budget": budget,
        "calls_remaining": max(0, budget - used),
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def _parse_pollen_response(raw):
    """Reshape Atmospore's /v1/pollen response into what the UI needs:
    per-day overall risk + tree/grass/weed totals, and a top-species
    leaderboard across the whole forecast window -- all derived from this
    ONE response so we never need the separate /v1/pollen-top or
    /v1/pollen-area endpoints (and their extra call cost) just for this tab.
    """
    days = raw.get("data", [])
    daily = []
    species_best = {}  # species_key -> best (highest-value) record seen

    for day in days:
        cat_totals = {"tree": 0.0, "grass": 0.0, "weed": 0.0}
        for species_key, info in day.get("species", {}).items():
            cat = info.get("category")
            val = info.get("value") or 0
            if cat in cat_totals:
                cat_totals[cat] += val
            prev = species_best.get(species_key)
            if prev is None or val > prev["value"]:
                species_best[species_key] = {
                    "species": species_key,
                    "display_name": info.get("display_name", species_key),
                    "category": cat,
                    "value": val,
                    "risk_level": info.get("risk_level", "low"),
                }
        daily.append({
            "date": day.get("date"),
            "overall_risk": day.get("overall_risk", "low"),
            "tree": cat_totals["tree"],
            "grass": cat_totals["grass"],
            "weed": cat_totals["weed"],
        })

    top_species = sorted(species_best.values(), key=lambda s: s["value"], reverse=True)[:5]

    return {
        "generated_at": raw.get("meta", {}).get("generated_at"),
        "units": raw.get("meta", {}).get("units", "grains/m\u00b3"),
        "daily": daily,
        "top_species": top_species,
    }


def _fetch_live(lat, lon, forecast_days):
    params = {
        "lat": lat,
        "lon": lon,
        "dt": _today_str(),
        "forecast_days": forecast_days,
        "species": "all",
    }
    headers = {"accept": "application/json"}
    if config.ATMOSPORE_API_KEY:
        headers["x-api-key"] = config.ATMOSPORE_API_KEY
    # No key configured -> Atmospore's own "demo mode" (per their docs: leave
    # the header empty to try without a key). Still worth attempting rather
    # than refusing locally, since demo mode may return sample data.

    resp = requests.get(
        config.ATMOSPORE_BASE_URL + "/pollen", params=params,
        headers=headers, timeout=15,
    )
    if resp.status_code == 429:
        raise _RateLimited(resp.text[:200])
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_pollen_forecast(city_slug, lat, lon, forecast_days=None, force_refresh=False):
    """Main entry point for the UI. Never raises -- always returns a status
    dict the caller can render directly:

        status: "live" | "cached" | "stale_cache" | "limit_reached" | "error"
        parsed: the reshaped forecast dict, or None
        fetched_at / cache_age_hours: when the shown data was actually fetched
        calls_used_today / calls_budget / calls_remaining
        message: human-readable explanation, safe to show in the UI
    """
    forecast_days = forecast_days or config.ATMOSPORE_FORECAST_DAYS
    today = _today_str()
    cache_path = _cache_path(city_slug)
    cached = _load_json(cache_path)

    # 1. Same-day cache already covers this -- reuse it, spend nothing.
    if cached and cached.get("fetched_date") == today and not force_refresh:
        return {
            "status": "cached",
            "parsed": cached["parsed"],
            "fetched_at": cached["fetched_at"],
            "cache_age_hours": _age_hours(cached["fetched_at"]),
            **get_call_budget_status(),
            "message": "Today's forecast was already fetched earlier today -- "
                       "reusing it instead of spending another call.",
        }

    budget = get_call_budget_status()
    if budget["calls_remaining"] <= 0:
        if cached:
            return {
                "status": "stale_cache",
                "parsed": cached["parsed"],
                "fetched_at": cached["fetched_at"],
                "cache_age_hours": _age_hours(cached["fetched_at"]),
                **budget,
                "message": f"Daily Atmospore call budget ({budget['calls_budget']}/day, "
                           "shared across every city) is used up for today. Showing "
                           "the last successfully fetched forecast instead.",
            }
        return {
            "status": "limit_reached",
            "parsed": None,
            "fetched_at": None,
            "cache_age_hours": None,
            **budget,
            "message": f"Daily Atmospore call budget ({budget['calls_budget']}/day) "
                       f"is used up, and no cached forecast exists yet for "
                       f"{city_slug.replace('_', ' ').title()}. Resets at 00:00 UTC.",
        }

    # 2. Budget available -- try exactly one live call (no retry loop: a
    # 429 here means "done for the day", not "wait a few seconds").
    try:
        raw = _fetch_live(lat, lon, forecast_days)
    except _RateLimited:
        _mark_ledger_exhausted()
        if cached:
            return {
                "status": "stale_cache",
                "parsed": cached["parsed"],
                "fetched_at": cached["fetched_at"],
                "cache_age_hours": _age_hours(cached["fetched_at"]),
                **get_call_budget_status(),
                "message": "The Atmospore API just reported its daily limit reached. "
                           "Showing the last cached forecast instead.",
            }
        return {
            "status": "limit_reached",
            "parsed": None,
            "fetched_at": None,
            "cache_age_hours": None,
            **get_call_budget_status(),
            "message": "The Atmospore API reported its daily limit reached, and no "
                       f"cached forecast exists yet for "
                       f"{city_slug.replace('_', ' ').title()}.",
        }
    except Exception as e:
        if cached:
            return {
                "status": "stale_cache",
                "parsed": cached["parsed"],
                "fetched_at": cached["fetched_at"],
                "cache_age_hours": _age_hours(cached["fetched_at"]),
                **get_call_budget_status(),
                "message": f"Live Atmospore request failed ({e}). Showing the last "
                           "cached forecast instead.",
            }
        return {
            "status": "error",
            "parsed": None,
            "fetched_at": None,
            "cache_age_hours": None,
            **get_call_budget_status(),
            "message": f"Live Atmospore request failed ({e}), and no cached "
                       f"forecast exists yet for "
                       f"{city_slug.replace('_', ' ').title()}.",
        }

    # 3. Success -- parse, cache, and count it against today's budget.
    parsed = _parse_pollen_response(raw)
    now_iso = dt.datetime.utcnow().isoformat()
    _save_json(cache_path, {
        "fetched_date": today,
        "fetched_at": now_iso,
        "parsed": parsed,
    })
    calls_used = _increment_ledger()

    return {
        "status": "live",
        "parsed": parsed,
        "fetched_at": now_iso,
        "cache_age_hours": 0.0,
        "calls_used_today": calls_used,
        "calls_budget": config.ATMOSPORE_DAILY_CALL_BUDGET,
        "calls_remaining": max(0, config.ATMOSPORE_DAILY_CALL_BUDGET - calls_used),
        "message": "Fetched fresh from the Atmospore API.",
    }
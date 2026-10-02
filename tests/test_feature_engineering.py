"""Tests for the feature builder.

These cover the two defects that changed what the model was actually learning:
rolling averages of the pollutants included the current hour (from which the
label is computed), and rolling/lag windows counted rows rather than hours on
a timeline with 40% of its hours missing.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_lag_features, add_rolling_features, add_time_features,
    hour_to_bucket, month_to_season,
)

GAP_START, GAP_HOURS = 150, 100


def _series_with_a_gap():
    """300 hourly rows with hours 150-249 removed, values = row position."""
    full = pd.date_range("2026-01-01", periods=300, freq="h")
    kept = list(full[:GAP_START]) + list(full[GAP_START + GAP_HOURS:])
    df = pd.DataFrame({"datetime": kept})
    df["pm25"] = np.arange(len(df), dtype=float)
    df["temperature_2m"] = np.arange(len(df), dtype=float) * 2
    return df


def _contiguous():
    df = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=300, freq="h")})
    df["pm25"] = np.arange(len(df), dtype=float)
    df["temperature_2m"] = np.arange(len(df), dtype=float) * 2
    return df


class TestTargetLeakage:
    def test_shifted_rolling_mean_excludes_the_current_hour(self):
        """risk_category comes from the current hour's pm25, so a rolling mean
        that includes that hour carries the answer."""
        df = _contiguous()
        out = add_rolling_features(df, ["pm25"], windows_hours=(24,),
                                   shift_hours=1).set_index("datetime")
        t = pd.Timestamp("2026-01-06 00:00")
        raw = df.set_index("datetime")["pm25"]
        expected = raw.loc[t - pd.Timedelta(hours=24):t - pd.Timedelta(hours=1)].mean()
        inclusive = raw.loc[t - pd.Timedelta(hours=23):t].mean()

        assert out.loc[t, "pm25_roll_24h_mean"] == pytest.approx(expected)
        assert out.loc[t, "pm25_roll_24h_mean"] != pytest.approx(inclusive)

    def test_unshifted_rolling_mean_includes_it(self):
        """Weather is exogenous, so it may legitimately use the current hour."""
        df = _contiguous()
        out = add_rolling_features(df, ["temperature_2m"], windows_hours=(24,),
                                   shift_hours=0).set_index("datetime")
        t = pd.Timestamp("2026-01-06 00:00")
        raw = df.set_index("datetime")["temperature_2m"]
        expected = raw.loc[t - pd.Timedelta(hours=23):t].mean()
        assert out.loc[t, "temperature_2m_roll_24h_mean"] == pytest.approx(expected)


class TestWindowsAreHoursNotRows:
    def test_rolling_window_does_not_bridge_a_gap(self):
        """A 24h window immediately after a 100h hole has almost no data in it,
        so it must be NaN rather than an average of whatever rows adjoin."""
        out = add_rolling_features(_series_with_a_gap(), ["pm25"],
                                   windows_hours=(24,), shift_hours=1)
        out = out.set_index("datetime")
        first_after_gap = out.index[GAP_START]
        assert np.isnan(out.loc[first_after_gap, "pm25_roll_24h_mean"])

    def test_lag_is_exactly_24_hours_or_nothing(self):
        out = add_lag_features(_series_with_a_gap(), ["pm25"], lags_hours=(24,))
        present = set(out["datetime"])
        raw = out.set_index("datetime")["pm25"]
        for t, lagged in out.set_index("datetime")["pm25_lag_24h"].items():
            source = t - pd.Timedelta(hours=24)
            if pd.isna(lagged):
                assert source not in present
            else:
                assert source in present
                assert lagged == pytest.approx(raw.loc[source])

    def test_feature_rows_line_up_with_the_input_rows(self):
        df = _series_with_a_gap()
        out = add_rolling_features(df, ["pm25"], windows_hours=(24,))
        assert len(out) == len(df)
        assert list(out["datetime"]) == list(df["datetime"])
        assert out["pm25"].equals(df["pm25"])


class TestSeasonsAndClock:
    @pytest.mark.parametrize("month,north,south", [
        (1, "winter", "summer"),
        (7, "summer", "winter"),
        (4, "spring", "autumn"),
        (10, "autumn", "spring"),
    ])
    def test_seasons_flip_below_the_equator(self, month, north, south):
        assert month_to_season(month) == north
        assert month_to_season(month, southern_hemisphere=True) == south

    def test_rejects_a_non_month(self):
        with pytest.raises(ValueError):
            month_to_season(13)

    def test_clock_features_come_from_local_time(self):
        """A UTC midnight is 05:30 in Delhi; hour must say 5, not 0."""
        df = pd.DataFrame({"datetime": pd.to_datetime(["2026-01-01 00:00"])})
        out = add_time_features(df, tz="Asia/Kolkata")
        assert out["hour"].iloc[0] == 5
        assert out["time_of_day"].iloc[0] == "morning"

    def test_local_conversion_follows_dst(self):
        """Auckland is UTC+13 in January and UTC+12 in June."""
        df = pd.DataFrame({"datetime": pd.to_datetime(
            ["2026-01-15 00:00", "2026-06-15 00:00"])})
        out = add_time_features(df, tz="Pacific/Auckland")
        assert out["hour"].tolist() == [13, 12]

    def test_dummy_columns_exist_for_every_level(self):
        """A 3-day forecast window sees one season and maybe two time buckets;
        the column set still has to match what training produced."""
        df = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=3, freq="h")})
        out = add_time_features(df, tz="UTC")
        for season in ("winter", "spring", "summer", "autumn"):
            assert f"season_{season}" in out.columns
        for bucket in ("morning", "afternoon", "evening", "night"):
            assert f"time_of_day_{bucket}" in out.columns

    def test_datetime_column_stays_utc(self):
        """The join key must not be rewritten to local time."""
        df = pd.DataFrame({"datetime": pd.to_datetime(["2026-01-01 00:00"])})
        out = add_time_features(df, tz="Asia/Kolkata")
        assert out["datetime"].iloc[0] == pd.Timestamp("2026-01-01 00:00")
        assert out["datetime_local"].iloc[0] == pd.Timestamp("2026-01-01 05:30")

    @pytest.mark.parametrize("hour,bucket", [
        (6, "morning"), (13, "afternoon"), (19, "evening"), (2, "night"),
    ])
    def test_hour_buckets(self, hour, bucket):
        assert hour_to_bucket(hour) == bucket

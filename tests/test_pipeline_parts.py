"""Tests for the OpenAQ paginator, the data-derived split, and the XGBoost
label wrapper.

The paginator test is the important one: the break condition it pins down was
dropping roughly 40% of the collected history, in contiguous blocks, without
raising anything.
"""

import numpy as np
import pandas as pd
import pytest

from src import fetch_air_quality
from src.model_wrappers import ContiguousLabelClassifier
from src.utils import chunk_date_range, derive_split_date, to_local


class TestOpenAQPagination:
    """The v3 API returns meta.found as the STRING '>1000' once the result set
    exceeds one page. The old break condition read that as 0 and stopped after
    page 1, keeping only the first 1000 hours of every 2160-hour chunk.
    """

    @pytest.fixture(autouse=True)
    def _no_network_no_sleeping(self, monkeypatch):
        monkeypatch.setattr(fetch_air_quality.config,
                            "OPENAQ_REQUEST_DELAY_SECONDS", 0)
        monkeypatch.setattr(fetch_air_quality, "_headers", lambda: {})

    def _install_fake_api(self, monkeypatch, total_rows, found_value):
        """A sensor holding total_rows records, served 1000 at a time."""
        limit = fetch_air_quality.PAGE_LIMIT
        calls = []

        def fake_get(url, params=None, headers=None, **kwargs):
            calls.append(params["page"])
            page = params["page"]
            start = (page - 1) * limit
            n = max(0, min(limit, total_rows - start))
            return {
                "results": [
                    {"period": {"datetimeFrom": {"utc": f"2026-01-01T{i % 24:02d}:00:00Z"}},
                     "value": float(i)}
                    for i in range(start, start + n)
                ],
                "meta": {"found": found_value},
            }

        monkeypatch.setattr(fetch_air_quality, "get_json_with_retries", fake_get)
        return calls

    def test_paginates_past_the_first_page_when_found_is_a_string(self, monkeypatch):
        calls = self._install_fake_api(monkeypatch, total_rows=2160, found_value=">1000")
        rows = fetch_air_quality.fetch_sensor_measurements(
            1, pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-01").date()
        )
        assert len(rows) == 2160, "stopped early -- the meta.found bug is back"
        assert calls == [1, 2, 3]

    def test_stops_on_a_short_page(self, monkeypatch):
        calls = self._install_fake_api(monkeypatch, total_rows=1500, found_value=">1000")
        rows = fetch_air_quality.fetch_sensor_measurements(
            1, pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-01").date()
        )
        assert len(rows) == 1500
        assert calls == [1, 2], "should not request a page beyond the last short one"

    def test_single_page_result_makes_exactly_one_request(self, monkeypatch):
        calls = self._install_fake_api(monkeypatch, total_rows=400, found_value=400)
        rows = fetch_air_quality.fetch_sensor_measurements(
            1, pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-05").date()
        )
        assert len(rows) == 400
        assert calls == [1]

    def test_has_a_runaway_page_cap(self, monkeypatch):
        """A server that never returns a short page must not loop forever."""
        calls = self._install_fake_api(monkeypatch, total_rows=10 ** 9, found_value=">1000")
        fetch_air_quality.fetch_sensor_measurements(
            1, pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-05").date()
        )
        assert max(calls) == fetch_air_quality.MAX_PAGES_PER_CHUNK


class TestChunking:
    def test_chunks_cover_the_range_without_overlap(self):
        start, end = pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-12-31").date()
        chunks = list(chunk_date_range(start, end, chunk_days=90))
        assert chunks[0][0] == start
        assert chunks[-1][1] == end
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
            assert (next_start - prev_end).days == 1


class TestDerivedSplitDate:
    def test_split_is_relative_to_the_data_not_today(self):
        """The old split was date.today() - 97 days, so it drifted off the end
        of the data and left the test set empty."""
        idx = pd.date_range("2025-07-31", "2026-07-26", freq="h")
        split = derive_split_date(idx, test_months=3)
        assert split == idx.max() - pd.Timedelta(days=90)
        assert idx.min() < split < idx.max()

    def test_short_datasets_keep_something_to_train_on(self):
        idx = pd.date_range("2026-07-01", "2026-07-26", freq="h")
        split = derive_split_date(idx, test_months=3)
        assert split > idx.min()
        assert (idx < split).sum() > 0
        assert (idx >= split).sum() > 0

    def test_rejects_an_empty_series(self):
        with pytest.raises(ValueError):
            derive_split_date([], test_months=3)


class TestToLocal:
    def test_fixed_offset_zone(self):
        out = to_local(pd.Series(pd.to_datetime(["2026-01-01 00:00"])), "Asia/Kolkata")
        assert out.iloc[0] == pd.Timestamp("2026-01-01 05:30")

    def test_dst_aware_zone(self):
        out = to_local(pd.Series(pd.to_datetime(
            ["2026-01-15 00:00", "2026-06-15 00:00"])), "Pacific/Auckland")
        assert out.iloc[0].hour == 13
        assert out.iloc[1].hour == 12

    def test_utc_is_a_no_op(self):
        stamps = pd.Series(pd.to_datetime(["2026-01-01 00:00"]))
        assert to_local(stamps, "UTC").iloc[0] == stamps.iloc[0]


class TestContiguousLabelClassifier:
    """The label encoder spans all six risk categories in severity order, so a
    train split can legitimately be missing one -- which XGBoost >= 1.7 rejects
    outright ("Invalid classes inferred from unique values of y").
    """

    def test_fits_and_predicts_with_a_hole_in_the_label_set(self):
        from xgboost import XGBClassifier

        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 4))
        y = np.array([0, 1, 2, 4, 5] * 24)          # 3 is absent -- not contiguous
        X[:, 0] = y                                  # make it trivially learnable

        clf = ContiguousLabelClassifier(
            XGBClassifier(n_estimators=10, max_depth=2, eval_metric="mlogloss")
        ).fit(X, y)
        preds = clf.predict(X)

        assert set(np.unique(preds)).issubset({0, 1, 2, 4, 5})
        assert 3 not in set(np.unique(preds))
        assert (preds == y).mean() > 0.9, "labels were not mapped back correctly"

    def test_probabilities_are_widened_to_the_canonical_label_space(self):
        from xgboost import XGBClassifier

        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 3))
        y = np.array([0, 2, 5] * 20)
        clf = ContiguousLabelClassifier(
            XGBClassifier(n_estimators=10, max_depth=2, eval_metric="mlogloss")
        ).fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape[1] == 6
        # Unseen classes get a hard zero, so column i always means class i.
        assert proba[:, [1, 3, 4]].sum() == pytest.approx(0.0)
        assert proba.sum(axis=1) == pytest.approx(np.ones(len(X)), abs=1e-5)

    def test_exposes_the_inner_model_for_shap(self):
        from xgboost import XGBClassifier

        inner = XGBClassifier(n_estimators=5, eval_metric="mlogloss")
        clf = ContiguousLabelClassifier(inner)
        assert clf.inner_model is inner


class TestReadPipelineCsv:
    """The app reads features.csv and merged_clean.csv with one reader.

    features.csv has both datetime and datetime_local; merged_clean.csv has
    only datetime. Parsing just the first left datetime_local as strings, and
    .min().to_pydatetime() on it raised AttributeError in the date picker.
    """

    def _write(self, tmp_path, cols):
        import pandas as pd

        stamps = pd.date_range("2026-01-01", periods=5, freq="h")
        data = {"datetime": stamps}
        if "datetime_local" in cols:
            data["datetime_local"] = stamps + pd.Timedelta(hours=5, minutes=30)
        data["pm25"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        path = tmp_path / "f.csv"
        pd.DataFrame(data).to_csv(path, index=False)
        return path

    def test_parses_both_datetime_columns(self, tmp_path):
        from src.utils import read_pipeline_csv

        df = read_pipeline_csv(self._write(tmp_path, ["datetime", "datetime_local"]))
        assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
        assert pd.api.types.is_datetime64_any_dtype(df["datetime_local"])
        # The call that actually crashed the app.
        assert df["datetime_local"].min().to_pydatetime().hour == 5

    def test_works_when_datetime_local_is_absent(self, tmp_path):
        from src.utils import read_pipeline_csv

        df = read_pipeline_csv(self._write(tmp_path, ["datetime"]))
        assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
        assert "datetime_local" not in df.columns

    def test_round_trips_what_the_feature_builder_writes(self, tmp_path):
        """Guards the real path: build features, save, reload, use."""
        from src.feature_engineering import add_time_features
        from src.utils import read_pipeline_csv

        df = pd.DataFrame({"datetime": pd.date_range("2026-01-01", periods=30, freq="h")})
        built = add_time_features(df, tz="Asia/Kolkata")
        path = tmp_path / "features.csv"
        built.to_csv(path, index=False)

        reloaded = read_pipeline_csv(path)
        assert pd.api.types.is_datetime64_any_dtype(reloaded["datetime_local"])
        assert reloaded["datetime_local"].min().to_pydatetime() == \
            built["datetime_local"].min().to_pydatetime()


class TestStationSelection:
    """Slots must go to stations that are still reporting.

    Delhi has 96 stations within 25km and only 6 slots. Sorting by distance
    alone gave three of them to stations whose last readings were in 2016,
    2018 and 2018, so half the request budget returned nothing while ~90 live
    stations slightly further out were never queried.
    """

    def _loc(self, name, lat, lon, last):
        rec = {"name": name, "coordinates": {"latitude": lat, "longitude": lon},
               "sensors": [{"id": 1, "parameter": {"name": "pm25"}}]}
        if last is not None:
            rec["datetimeLast"] = {"utc": last}
        return rec

    def _install(self, monkeypatch, locations):
        from src import fetch_air_quality
        monkeypatch.setattr(fetch_air_quality, "_headers", lambda: {})
        monkeypatch.setattr(fetch_air_quality, "get_json_with_retries",
                            lambda *a, **k: {"results": locations})

    def test_retired_stations_lose_their_slot_to_live_ones(self, monkeypatch):
        import datetime as dt
        from src.fetch_air_quality import find_nearby_locations

        self._install(monkeypatch, [
            self._loc("dead-and-close", 0.001, 0.0, "2018-02-21T00:00:00Z"),
            self._loc("dead-and-closer", 0.0005, 0.0, "2016-11-09T00:00:00Z"),
            self._loc("live-but-further", 0.05, 0.0, "2026-10-02T00:00:00Z"),
        ])
        picked = find_nearby_locations(0.0, 0.0, 25000, max_locations=2,
                                       active_since=dt.date(2025, 9, 30))
        names = [p["name"] for p in picked]
        assert names[0] == "live-but-further"
        assert "dead-and-closer" not in names

    def test_distance_still_decides_among_live_stations(self, monkeypatch):
        import datetime as dt
        from src.fetch_air_quality import find_nearby_locations

        self._install(monkeypatch, [
            self._loc("far", 0.10, 0.0, "2026-10-02T00:00:00Z"),
            self._loc("near", 0.01, 0.0, "2026-10-02T00:00:00Z"),
        ])
        picked = find_nearby_locations(0.0, 0.0, 25000, max_locations=2,
                                       active_since=dt.date(2025, 9, 30))
        assert [p["name"] for p in picked] == ["near", "far"]

    def test_stations_with_no_date_metadata_rank_behind_confirmed_live(self, monkeypatch):
        import datetime as dt
        from src.fetch_air_quality import find_nearby_locations

        self._install(monkeypatch, [
            self._loc("unknown-and-close", 0.001, 0.0, None),
            self._loc("live-and-far", 0.10, 0.0, "2026-10-02T00:00:00Z"),
        ])
        picked = find_nearby_locations(0.0, 0.0, 25000, max_locations=2,
                                       active_since=dt.date(2025, 9, 30))
        # Kept (we cannot prove it is dead) but not ahead of a confirmed one.
        assert [p["name"] for p in picked] == ["live-and-far", "unknown-and-close"]

    def test_without_a_window_behaviour_is_pure_distance(self, monkeypatch):
        from src.fetch_air_quality import find_nearby_locations

        self._install(monkeypatch, [
            self._loc("dead-and-close", 0.001, 0.0, "2016-01-01T00:00:00Z"),
            self._loc("live-and-far", 0.10, 0.0, "2026-10-02T00:00:00Z"),
        ])
        picked = find_nearby_locations(0.0, 0.0, 25000, max_locations=2)
        assert [p["name"] for p in picked] == ["dead-and-close", "live-and-far"]


class TestGroundCoverageAssessment:
    """Finding a station is not the same as getting usable data from it."""

    START, END = "2025-09-30", "2026-09-25"

    def _frame(self, start, hours):
        return pd.DataFrame({
            "datetime": pd.date_range(start, periods=hours, freq="h"),
            "pm25": [10.0] * hours,
        })

    def test_accepts_a_well_covered_recent_series(self):
        from src.fetch_air_quality import assess_ground_coverage
        import datetime as dt

        df = self._frame("2025-09-30", 8600)
        usable, reason = assess_ground_coverage(
            df, dt.date(2025, 9, 30), dt.date(2026, 9, 25))
        assert usable, reason

    def test_rejects_a_series_that_stops_two_months_in(self):
        """The Hamilton case: both stations died 2025-12-03."""
        from src.fetch_air_quality import assess_ground_coverage
        import datetime as dt

        df = self._frame("2025-09-30", 1553)
        usable, reason = assess_ground_coverage(
            df, dt.date(2025, 9, 30), dt.date(2026, 9, 25))
        assert not usable
        assert "18%" in reason

    def test_rejects_good_coverage_that_ends_long_ago(self):
        """Catches a station that reported well, then died."""
        from src.fetch_air_quality import assess_ground_coverage
        import datetime as dt

        # 70% of the window, but all of it finishing ~4 months early.
        df = self._frame("2025-09-30", 6100)
        usable, reason = assess_ground_coverage(
            df, dt.date(2025, 9, 30), dt.date(2026, 9, 25))
        assert not usable
        assert "days before the end" in reason

    def test_rejects_nothing_at_all(self):
        from src.fetch_air_quality import assess_ground_coverage
        import datetime as dt

        for empty in (None, pd.DataFrame()):
            usable, _ = assess_ground_coverage(
                empty, dt.date(2025, 9, 30), dt.date(2026, 9, 25))
            assert not usable

"""Tests for the AQI breakpoint math and the risk-category label space.

labeling.py is pure and deterministic, which makes it the cheapest thing in
the project to pin down -- and it is where two real bugs lived: concentrations
falling between printed breakpoint bands returned NaN, and an AQI falling
between printed category bands was labelled Hazardous.
"""

import numpy as np
import pytest

from src.labeling import (
    PM10_BREAKPOINTS, PM25_BREAKPOINTS_2012, PM25_BREAKPOINTS_2024,
    NO2_BREAKPOINTS_PPB, RISK_CATEGORY_ORDER, OrderedLabelEncoder,
    _sub_index, aqi_to_category, compute_aqi_row, pm25_breakpoints,
)


@pytest.mark.parametrize("breakpoints,decimals,hi", [
    (PM25_BREAKPOINTS_2024, 1, 400),
    (PM25_BREAKPOINTS_2012, 1, 600),
    (PM10_BREAKPOINTS, 0, 700),
    (NO2_BREAKPOINTS_PPB, 0, 2200),
])
def test_no_concentration_falls_into_a_breakpoint_gap(breakpoints, decimals, hi):
    """Every non-negative concentration must produce a usable sub-index.

    The tables are printed with gaps (PM2.5 runs 0.0-12.0 then 12.1-35.4), so
    a value of 12.05 matched no band and returned NaN -- which silently dropped
    that pollutant out of the overall max() and understated the hour's AQI.
    """
    step = 10 ** -decimals / 2 if decimals else 0.5
    offenders = [v for v in np.arange(0, hi, step)
                 if np.isnan(_sub_index(v, breakpoints, decimals))]
    assert offenders == []


def test_sub_index_is_monotonic_in_concentration():
    values = np.arange(0, 400, 0.1)
    indices = [_sub_index(v, pm25_breakpoints("2024"), 1) for v in values]
    assert all(b >= a - 1e-9 for a, b in zip(indices, indices[1:]))


def test_sub_index_rejects_only_genuinely_bad_input():
    assert np.isnan(_sub_index(-1, pm25_breakpoints(), 1))
    assert np.isnan(_sub_index(float("nan"), pm25_breakpoints(), 1))
    assert np.isnan(_sub_index(None, pm25_breakpoints(), 1))


def test_sub_index_caps_at_500_above_the_scale():
    assert _sub_index(10_000, pm25_breakpoints(), 1) == 500.0


def test_2024_table_moved_the_good_moderate_boundary():
    """EPA's 2024 revision dropped the Good ceiling from 12.0 to 9.0 ug/m3."""
    assert _sub_index(10.0, PM25_BREAKPOINTS_2024, 1) > 50
    assert _sub_index(10.0, PM25_BREAKPOINTS_2012, 1) <= 50
    assert _sub_index(9.0, PM25_BREAKPOINTS_2024, 1) == pytest.approx(50)


def test_pm25_breakpoints_rejects_an_unknown_version():
    with pytest.raises(ValueError):
        pm25_breakpoints("1999")


@pytest.mark.parametrize("aqi,expected", [
    (0, "Good"),
    (50, "Good"),
    (50.4, "Good"),       # used to return Hazardous: fell in the 50/51 gap
    (100.4, "Moderate"),  # same, in the 100/101 gap
    (150.7, "Unhealthy"),
    (200.2, "Unhealthy"),
    (250, "Very Unhealthy"),
    (301, "Hazardous"),
    (99_999, "Hazardous"),
])
def test_aqi_to_category_covers_the_gaps_between_bands(aqi, expected):
    assert aqi_to_category(aqi) == expected


def test_aqi_to_category_passes_through_unusable_input():
    assert aqi_to_category(None) is np.nan or np.isnan(aqi_to_category(None))
    assert np.isnan(aqi_to_category(float("nan")))
    assert np.isnan(aqi_to_category(-5))


def test_overall_aqi_is_the_max_of_the_sub_indices():
    pm25_only = compute_aqi_row(pm25=150.0)
    both = compute_aqi_row(pm25=150.0, pm10=20.0)
    assert both == pm25_only


def test_a_missing_pollutant_does_not_sink_the_overall_aqi():
    assert compute_aqi_row(pm25=150.0, pm10=None, no2_ugm3=float("nan")) == \
        compute_aqi_row(pm25=150.0)


class TestOrderedLabelEncoder:
    def test_codes_follow_severity_not_the_alphabet(self):
        le = OrderedLabelEncoder().fit(["Good", "Hazardous"])
        # sklearn's LabelEncoder would give Good=0, Hazardous=1, Moderate=2
        assert list(le.classes_) == RISK_CATEGORY_ORDER
        assert le.transform(["Good"])[0] == 0
        assert le.transform(["Moderate"])[0] == 1
        assert le.transform(["Hazardous"])[0] == len(RISK_CATEGORY_ORDER) - 1

    def test_class_codes_are_stable_across_differing_datasets(self):
        """A city with only Good data must still agree on what code 3 means."""
        clean = OrderedLabelEncoder().fit(["Good"])
        dirty = OrderedLabelEncoder().fit(["Moderate", "Unhealthy", "Hazardous"])
        assert list(clean.classes_) == list(dirty.classes_)

    def test_round_trips(self):
        le = OrderedLabelEncoder().fit(RISK_CATEGORY_ORDER)
        codes = le.transform(RISK_CATEGORY_ORDER)
        assert list(le.inverse_transform(codes)) == RISK_CATEGORY_ORDER

    def test_rejects_an_unknown_label(self):
        with pytest.raises(ValueError):
            OrderedLabelEncoder().fit(["Good", "Apocalyptic"])

    def test_rejects_an_out_of_range_code(self):
        le = OrderedLabelEncoder().fit(["Good"])
        with pytest.raises(ValueError):
            le.inverse_transform([99])

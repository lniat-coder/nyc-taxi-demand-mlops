"""Data quality checks: tell an upstream bug apart from real drift.

A broken pipeline looks exactly like drift (course, session 5). Before any
retraining decision, these checks look for patterns that real demand does not
produce: a busy zone reporting exactly zero for a whole day, or an hour with
no trip at all in the whole city.
"""

from __future__ import annotations

import pandas as pd


def longest_zero_run(values: pd.Series) -> int:
    """Length of the longest run of consecutive zeros."""
    is_zero = values.eq(0)
    run_id = (~is_zero).cumsum()
    runs = is_zero.groupby(run_id).sum()
    return int(runs.max()) if len(runs) else 0


def check_window(window: pd.DataFrame, reference_zone_means: dict[int, float],
                 silent_zone_hours: int, silent_zone_min_ref_mean: float,
                 max_empty_hours: int) -> list[dict]:
    """Return a list of issues found in a window of (zone_id, hour_start, pickups) rows.

    A zone is "silent" only if it should still be busy *given the city's current volume*:
    its reference mean is scaled by (city pickups per hour now / in the reference period).
    Without that scaling, the real April 2020 collapse (-96% city-wide) made small zones
    look broken (zones 45 and 209 in our first run on the real data).
    """
    issues: list[dict] = []
    ref_city = sum(reference_zone_means.values())
    city_scale = window.groupby("hour_start")["pickups"].sum().mean() / ref_city if ref_city else 1.0

    for zone_id, rows in window.sort_values("hour_start").groupby("zone_id"):
        ref_mean = reference_zone_means.get(int(zone_id), 0.0)
        expected = ref_mean * city_scale
        if expected < silent_zone_min_ref_mean:
            continue
        run = longest_zero_run(rows["pickups"])
        if run >= silent_zone_hours:
            issues.append({"check": "silent_zone", "zone_id": int(zone_id), "zero_hours": run,
                           "reference_mean_per_hour": round(ref_mean, 1),
                           "expected_mean_per_hour_now": round(expected, 1)})

    per_hour = window.groupby("hour_start")["pickups"].sum()
    empty = int((per_hour == 0).sum())
    if empty > max_empty_hours:
        issues.append({"check": "empty_hours", "count": empty})

    if window[["zone_id", "hour_start"]].duplicated().any():
        issues.append({"check": "duplicate_zone_hour"})
    if (window["pickups"] < 0).any():
        issues.append({"check": "negative_pickups"})
    return issues

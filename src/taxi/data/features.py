"""(zone, hour) counts -> model features.

Task: predict the pickups of a zone for hour t from the hours before t
(lags >= 1) and calendar features of t. Every lag is computed inside its own
zone and only looks backwards, so there is no leakage from the future.
"""

from __future__ import annotations

import logging

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from taxi.config import FEATURES_PATH, ZONE_HOUR_PATH, params

log = logging.getLogger(__name__)


def build_features(zone_hour: pd.DataFrame, lags: list[int] | None = None,
                   rolling_window: int | None = None) -> pd.DataFrame:
    cfg = params()["features"]
    lags = lags or cfg["lags"]
    rolling_window = rolling_window or cfg["rolling_window"]

    df = zone_hour.sort_values(["zone_id", "hour_start"]).reset_index(drop=True)
    by_zone = df.groupby("zone_id")["pickups"]
    for lag in lags:
        df[f"lag_{lag}"] = by_zone.shift(lag)
    # mean of the 24 hours before t (shift(1) first, so hour t itself is excluded)
    df[f"roll_mean_{rolling_window}"] = by_zone.transform(
        lambda s: s.shift(1).rolling(rolling_window).mean())

    df = pd.concat([df, calendar_features(df["hour_start"])], axis=1)
    # The first 168 hours of each zone have no full history: drop them.
    return df.dropna().reset_index(drop=True)


def calendar_features(ts: pd.Series) -> pd.DataFrame:
    """Calendar features of the forecast hour. Also used by the API, so train and serve match."""
    ts = pd.to_datetime(ts)
    holidays = USFederalHolidayCalendar().holidays(ts.min().normalize(), ts.max().normalize())
    dow = ts.dt.dayofweek
    return pd.DataFrame({
        "hour": ts.dt.hour.astype("int8"),
        "day_of_week": dow.astype("int8"),
        "is_weekend": (dow >= 5).astype("int8"),
        "is_holiday": ts.dt.normalize().isin(holidays).astype("int8"),
    }, index=ts.index)


def time_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows whose hour falls between two dates, both included (split by time, never at random)."""
    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df.hour_start >= pd.Timestamp(start)) & (df.hour_start < end_excl)]


def main() -> None:
    zone_hour = pd.read_parquet(ZONE_HOUR_PATH)
    features = build_features(zone_hour)
    features.to_parquet(FEATURES_PATH, index=False)
    log.info("wrote %s: %d rows, %s -> %s", FEATURES_PATH, len(features),
             features.hour_start.min(), features.hour_start.max())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

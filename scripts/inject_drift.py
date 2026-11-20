"""Inject an upstream data incident into a copy of the data (the original stays untouched).

zone_silent    a zone stops reporting trips (pickups forced to 0) between two dates, as if
               its records were dropped by a broken ingestion job. The features are rebuilt
               from the corrupted counts, so the lags see the bug as they would in production.
               Expected reaction: the quality check says FIX_PIPELINE, nothing is retrained.

leaky_feature  a refactor of the feature pipeline forgets the shift: lag_1 becomes the pickups of
               the hour to predict (off-by-one leakage). Validation looks excellent because the
               holdout has the same leak, so the gate passes. The API receives the real lag_1,
               so the model is wrong in production. Expected reaction: caught on fresh labels,
               roll back.

    python scripts/inject_drift.py --mode zone_silent --zone 237 --start 2020-05-01 --end 2020-05-31
    python scripts/inject_drift.py --mode leaky_feature --start 2020-04-15 --end 2020-05-31
    -> data/injected/features_<mode>.parquet
"""

from __future__ import annotations

import argparse

import pandas as pd

from taxi.config import FEATURES_PATH, INJECTED_DIR, ZONE_HOUR_PATH
from taxi.data.features import build_features


def zone_silent(zone_hour: pd.DataFrame, zone: int, start: str, end: str) -> pd.DataFrame:
    out = zone_hour.copy()
    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    mask = (out.zone_id == zone) & (out.hour_start >= pd.Timestamp(start)) & (out.hour_start < end_excl)
    if not mask.any():
        raise SystemExit(f"zone {zone} has no rows between {start} and {end}")
    out.loc[mask, "pickups"] = 0
    print(f"zone {zone}: {int(mask.sum())} hours silenced ({int(zone_hour.loc[mask, 'pickups'].sum()):,} pickups removed)")
    return out


def leaky_feature(features: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = features.copy()
    end_excl = pd.Timestamp(end) + pd.Timedelta(days=1)
    mask = (out.hour_start >= pd.Timestamp(start)) & (out.hour_start < end_excl)
    out.loc[mask, "lag_1"] = out.loc[mask, "pickups"]  # shift(1) forgotten: lag_1 is the target
    print(f"lag_1 now equals the target on {int(mask.sum()):,} rows ({start} .. {end})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["zone_silent", "leaky_feature"], default="zone_silent")
    ap.add_argument("--zone", type=int, default=237)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()

    if a.mode == "zone_silent":
        out = build_features(zone_silent(pd.read_parquet(ZONE_HOUR_PATH), a.zone, a.start, a.end))
    else:
        out = leaky_feature(pd.read_parquet(FEATURES_PATH), a.start, a.end)
    INJECTED_DIR.mkdir(parents=True, exist_ok=True)
    path = INJECTED_DIR / f"features_{a.mode}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

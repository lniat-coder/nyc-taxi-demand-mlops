"""Raw trips -> one row per (zone, hour) with the number of pickups.

Only two raw columns are read (pickup time and pickup zone), so a month of
~7 million trips fits in memory. The output is a complete grid: an hour with
no pickup in a served zone is a 0, not a missing row. That matters for
monitoring, because a zone that goes silent must show up as zeros.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from taxi.config import PROCESSED_DIR, RAW_DIR, ZONE_HOUR_PATH, ZONES_PATH, params

log = logging.getLogger(__name__)
COLUMNS = ["tpep_pickup_datetime", "PULocationID"]


def aggregate_month(path, month: str) -> pd.DataFrame:
    """Count pickups per zone and hour for one monthly file."""
    trips = pd.read_parquet(path, columns=COLUMNS)
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    ts = pd.to_datetime(trips["tpep_pickup_datetime"])
    in_month = (ts >= start) & (ts < end)  # TLC files contain a few trips dated 2008, 2088...
    dropped = int((~in_month).sum())
    if dropped:
        log.info("%s: dropped %d trips outside the month", month, dropped)
    counts = (
        pd.DataFrame({"hour_start": ts[in_month].dt.floor("h"),
                      "zone_id": trips.loc[in_month, "PULocationID"].astype("int16")})
        .groupby(["zone_id", "hour_start"]).size().rename("pickups").reset_index()
    )
    log.info("%s: %d trips -> %d zone-hours", month, int(in_month.sum()), len(counts))
    return counts


def select_zones(counts: pd.DataFrame, start: str, end: str, top_k: int, excluded: list[int]) -> list[int]:
    """Busiest zones of the training period only: no peek at the future."""
    window = counts[(counts.hour_start >= start) & (counts.hour_start < pd.Timestamp(end) + pd.Timedelta(days=1))]
    totals = window[~window.zone_id.isin(excluded)].groupby("zone_id")["pickups"].sum()
    return sorted(int(z) for z in totals.nlargest(top_k).index)


def complete_grid(counts: pd.DataFrame, zones: list[int], first_month: str, last_month: str) -> pd.DataFrame:
    """Every (zone, hour) of the period, zero-filled."""
    start = pd.Timestamp(f"{first_month}-01")
    end = pd.Timestamp(f"{last_month}-01") + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1)
    grid = pd.MultiIndex.from_product([zones, pd.date_range(start, end, freq="h")],
                                      names=["zone_id", "hour_start"])
    out = (counts[counts.zone_id.isin(zones)].set_index(["zone_id", "hour_start"])["pickups"]
           .reindex(grid, fill_value=0).reset_index())
    out["zone_id"] = out["zone_id"].astype("int16")
    out["pickups"] = out["pickups"].astype("int32")
    return out


def main() -> None:
    p = params()
    months = p["data"]["months"]
    counts = pd.concat([aggregate_month(RAW_DIR / f"yellow_tripdata_{m}.parquet", m) for m in months])
    counts = counts.groupby(["zone_id", "hour_start"], as_index=False)["pickups"].sum()

    zones = select_zones(counts, p["split"]["train_start"], p["split"]["train_end"],
                         p["data"]["top_k_zones"], p["data"]["excluded_zones"])
    zone_hour = complete_grid(counts, zones, months[0], months[-1])

    lookup = pd.read_csv(RAW_DIR / "taxi_zone_lookup.csv").set_index("LocationID")
    zone_info = {int(z): {"borough": str(lookup.at[z, "Borough"]), "zone": str(lookup.at[z, "Zone"])}
                 for z in zones}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    zone_hour.to_parquet(ZONE_HOUR_PATH, index=False)
    ZONES_PATH.write_text(json.dumps(zone_info, indent=2))
    log.info("wrote %s: %d rows, %d zones", ZONE_HOUR_PATH, len(zone_hour), len(zones))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

"""Replay a period of real (zone, hour) rows against the running API, as live traffic.

Rows are sampled at random from the period, so a window of requests covers all
hours of the day (otherwise the PSI would just see the daily cycle).

    python scripts/replay_traffic.py --start 2020-02-01 --end 2020-02-29   # normal traffic
    python scripts/replay_traffic.py --start 2020-04-01 --end 2020-04-30   # lockdown: drift alert
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

from taxi.config import API_URL, FEATURES_PATH
from taxi.data.features import time_slice

FIELDS = ["zone_id", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168", "roll_mean_24"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--features", type=Path, default=FEATURES_PATH)
    ap.add_argument("--n", type=int, default=2000, help="number of requests")
    ap.add_argument("--rps", type=float, default=25, help="requests per second")
    ap.add_argument("--url", default=API_URL)
    a = ap.parse_args()

    rows = time_slice(pd.read_parquet(a.features), a.start, a.end)
    rows = rows.sample(n=a.n, replace=len(rows) < a.n, random_state=0)
    statuses, versions = Counter(), Counter()
    session = requests.Session()
    t0 = time.perf_counter()
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        body = {k: (int(r[k]) if k == "zone_id" else float(r[k])) for k in FIELDS}
        body["hour_start"] = r["hour_start"].isoformat()
        resp = session.post(f"{a.url}/predict", json=body, timeout=5)
        statuses[resp.status_code] += 1
        if resp.ok:
            versions[resp.json()["model_version"]] += 1
        if i % 250 == 0:
            print(f"{i}/{a.n} sent, status {dict(statuses)}, model versions {dict(versions)}")
        time.sleep(max(0.0, i / a.rps - (time.perf_counter() - t0)))
    print(f"done: {dict(statuses)} in {time.perf_counter() - t0:.0f}s, served by versions {dict(versions)}")


if __name__ == "__main__":
    main()

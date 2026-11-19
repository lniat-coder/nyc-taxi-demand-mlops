"""Download the raw TLC yellow taxi files (one Parquet file per month).

Source: NYC Taxi & Limousine Commission (TLC) Trip Record Data,
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The raw files are never committed: this script (run by `dvc repro`) fetches
them again, and DVC records their md5 in dvc.lock.
"""

from __future__ import annotations

import logging
import time

import requests

from taxi.config import RAW_DIR, params

log = logging.getLogger(__name__)
RETRY_STATUS = {429, 500, 502, 503, 504}


def fetch(url: str, dest, retries: int = 4, timeout: int = 60) -> None:
    """Stream a file to disk, retrying on 429/5xx and network errors."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code in RETRY_STATUS:
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.rename(dest)  # atomic: a half-downloaded file never looks complete
            return
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status not in RETRY_STATUS:
                raise
            if attempt == retries:
                raise
            wait = 2**attempt
            log.warning("%s failed (%s), retry %d/%d in %ds", url, exc, attempt, retries, wait)
            time.sleep(wait)


def main() -> None:
    cfg = params()["data"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets = [(f"{cfg['base_url']}/yellow_tripdata_{m}.parquet", RAW_DIR / f"yellow_tripdata_{m}.parquet")
               for m in cfg["months"]]
    targets.append((cfg["zone_lookup_url"], RAW_DIR / "taxi_zone_lookup.csv"))
    for url, dest in targets:
        if dest.exists() and dest.stat().st_size > 0:
            log.info("already here: %s", dest.name)
            continue
        log.info("downloading %s", url)
        fetch(url, dest)
    log.info("raw data ready in %s", RAW_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

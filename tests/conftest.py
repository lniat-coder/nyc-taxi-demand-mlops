"""Shared fixtures. Tests use synthetic counts shaped like the TLC aggregates,
so they run without downloading anything (CI, fresh clone)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxi.config import params, policy
from taxi.data.features import build_features

ZONES = [48, 132, 161, 236, 237]


def synthetic_zone_hour(start="2019-11-01", end="2020-06-30 23:00", collapse_from="2020-03-16",
                        collapse_factor=0.1, seed=0) -> pd.DataFrame:
    """Daily and weekly cycles per zone, Poisson noise, demand x0.1 from mid-March 2020."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq="h")
    daily = 0.3 + np.sin(np.pi * (hours.hour.to_numpy() - 5) / 24).clip(0) ** 2 * 1.7
    weekly = np.where(hours.dayofweek.to_numpy() >= 5, 0.8, 1.0)
    shock = np.where(hours >= pd.Timestamp(collapse_from), collapse_factor, 1.0)
    frames = []
    for i, z in enumerate(ZONES):
        lam = (40 + 25 * i) * daily * weekly * shock
        frames.append(pd.DataFrame({"zone_id": np.int16(z), "hour_start": hours,
                                    "pickups": rng.poisson(lam).astype("int32")}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session", autouse=True)
def fast_settings():
    """Small model and small live PSI window so the tests stay fast."""
    params()["model"]["max_iter"] = 80
    params()["model"]["learning_rate"] = 0.2
    policy()["drift"]["live_window"] = 40
    policy()["drift"]["live_every"] = 10


@pytest.fixture(scope="session")
def zone_hour() -> pd.DataFrame:
    return synthetic_zone_hour()


@pytest.fixture(scope="session")
def features(zone_hour) -> pd.DataFrame:
    return build_features(zone_hour)

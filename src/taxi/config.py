"""Paths, parameters and environment settings shared by every module."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(os.environ.get("TAXI_ROOT", Path(__file__).resolve().parents[2]))
load_dotenv(ROOT / ".env")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INJECTED_DIR = DATA_DIR / "injected"
REPORTS_DIR = ROOT / "reports"

ZONE_HOUR_PATH = PROCESSED_DIR / "zone_hour.parquet"
ZONES_PATH = PROCESSED_DIR / "zones.json"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
API_URL = os.environ.get("API_URL", "http://localhost:8000")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")

TRAINING_EXPERIMENT = "nyc-taxi-demand-training"
MONITORING_EXPERIMENT = "nyc-taxi-demand-monitoring"


@lru_cache
def params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


@lru_cache
def policy() -> dict:
    with open(ROOT / "policy.yaml") as f:
        return yaml.safe_load(f)


def model_name() -> str:
    return os.environ.get("MODEL_NAME", params()["model"]["name"])


def feature_columns() -> list[str]:
    f = params()["features"]
    return f["categorical"] + f["numeric"]

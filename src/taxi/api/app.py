"""FastAPI service: serves models:/nyc-taxi-demand@production.

Endpoints
    GET  /health         200 when a model is loaded, 503 otherwise (readiness)
    GET  /model          which version is served, trained on what
    POST /predict        pickups forecast for one zone and one hour
    POST /admin/reload   reload @production (after a promotion or a rollback), needs X-Admin-Token
    GET  /metrics        Prometheus metrics (layers 1 to 3 of the course's monitoring)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from taxi.api.schemas import ModelInfo, PredictRequest, PredictResponse
from taxi.config import ADMIN_TOKEN, model_name, policy
from taxi.data.features import calendar_features
from taxi.model.core import alias_version, init_mlflow, load_version, to_model_input
from taxi.monitoring.psi import psi

log = logging.getLogger("taxi.api")

# ---- metrics: few labels, never an identifier (one label value = one time series) ----
LATENCY = Histogram("taxi_request_seconds", "Request latency", ["endpoint"],
                    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
REQUESTS = Counter("taxi_http_requests_total", "HTTP requests", ["endpoint", "status"])
PREDICTIONS = Counter("taxi_predictions_total", "Predictions served", ["model_version"])
PREDICTED = Histogram("taxi_predicted_pickups", "Predicted pickups per zone-hour", ["model_version"],
                      buckets=[0, 5, 10, 25, 50, 100, 200, 400, 800])
INPUT_PSI = Gauge("taxi_input_psi", "PSI of a feature: live window vs training data", ["feature"])
MODEL_LOADED = Gauge("taxi_model_loaded", "1 when a model is loaded")
MODEL_INFO = Gauge("taxi_model_version_info", "Model version currently served", ["model_version"])


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model = None
        self.version: str | None = None
        self.run_id: str | None = None
        self.profile: dict | None = None
        self.windows: dict[str, deque] = {}
        self.seen = 0


state = State()
_pred_log = logging.getLogger("taxi.predictions")


def _setup_prediction_log() -> None:
    """One JSON line per prediction (layer 4: joined later with the actual pickups)."""
    path = Path(os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl"))
    if not _pred_log.handlers:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(path)
        except OSError as exc:  # e.g. a mounted folder the container user cannot write
            log.warning("cannot write %s (%s): prediction log goes to stdout", path, exc)
            handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        _pred_log.addHandler(handler)
        _pred_log.setLevel(logging.INFO)
        _pred_log.propagate = False


def load_production() -> str:
    """Load the @production version and its reference profile, then swap it in."""
    init_mlflow()
    prod = alias_version(model_name(), "production")
    if prod is None:
        raise RuntimeError(f"no @production alias on model '{model_name()}'")
    mv, model, profile = load_version(model_name(), prod.version)
    window = policy()["drift"]["live_window"]
    with state.lock:
        old = state.version
        state.model, state.version, state.run_id, state.profile = model, str(mv.version), mv.run_id, profile
        state.windows = {f: deque(maxlen=window) for f in profile["psi"]}
        state.seen = 0
    if old is not None:
        MODEL_INFO.labels(old).set(0)
    MODEL_INFO.labels(state.version).set(1)
    MODEL_LOADED.set(1)
    for f in profile["psi"]:
        INPUT_PSI.labels(f).set(0)  # new reference: the gauge restarts once the window is full again
    log.info("serving %s version %s", model_name(), state.version)
    return state.version


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_prediction_log()
    MODEL_LOADED.set(0)
    try:
        load_production()  # once, at startup: loading takes seconds
    except Exception as exc:  # noqa: BLE001 - the API still starts, /health says 503
        log.warning("no model loaded at startup: %s", exc)
    yield


app = FastAPI(title="NYC taxi demand forecast", version="1.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    response = await call_next(request)
    seconds = time.perf_counter() - start
    path = request.url.path
    if not path.startswith("/metrics"):
        LATENCY.labels(path).observe(seconds)
        REQUESTS.labels(path, str(response.status_code)).inc()
    response.headers["X-Process-Time-Ms"] = f"{seconds * 1000:.2f}"
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health():
    if state.model is None:
        return JSONResponse({"status": "degraded", "detail": "no model loaded"}, status_code=503)
    return {"status": "ok", "model_version": state.version}


@app.get("/model", response_model=ModelInfo)
def model_info():
    if state.model is None:
        raise HTTPException(503, "no model loaded")
    p = state.profile
    return ModelInfo(model_name=model_name(), model_version=state.version, run_id=state.run_id,
                     train_window=p["train_window"], valid_wape=p["valid_wape"], n_zones=len(p["zones"]))


def _observe_inputs(row: dict) -> None:
    """Layer 2: sliding window per feature, PSI against training every `live_every` requests."""
    every = policy()["drift"]["live_every"]
    with state.lock:
        for f, window in state.windows.items():
            window.append(row[f])
        state.seen += 1
        due = state.seen % every == 0 and all(len(w) == w.maxlen for w in state.windows.values())
        snapshot = {f: np.array(w) for f, w in state.windows.items()} if due else None
        profile = state.profile
    if snapshot:
        for f, values in snapshot.items():
            INPUT_PSI.labels(f).set(psi(profile["psi"][f], values))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    model, version, profile = state.model, state.version, state.profile
    if model is None:
        raise HTTPException(503, "no model loaded")
    if req.zone_id not in profile["zones"]:
        raise HTTPException(422, f"zone {req.zone_id} is not served by model version {version}")

    row = req.model_dump()
    hour_start = pd.Timestamp(req.hour_start).tz_localize(None)
    frame = pd.DataFrame([row]).assign(hour_start=hour_start)
    frame = pd.concat([frame, calendar_features(frame["hour_start"])], axis=1)

    start = time.perf_counter()
    y_hat = max(float(model.predict(to_model_input(frame, profile["feature_columns"]))[0]), 0.0)
    model_ms = (time.perf_counter() - start) * 1000

    PREDICTIONS.labels(version).inc()
    PREDICTED.labels(version).observe(y_hat)
    _observe_inputs(row)
    _pred_log.info(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": request.state.request_id, "model_version": version,
        "zone_id": req.zone_id, "hour_start": hour_start.isoformat(), "predicted_pickups": round(y_hat, 3)}))
    return PredictResponse(request_id=request.state.request_id, zone_id=req.zone_id, hour_start=hour_start,
                           predicted_pickups=round(y_hat, 3), model_version=version, model_ms=round(model_ms, 3))


@app.post("/admin/reload")
def reload(x_admin_token: str | None = Header(default=None)):
    token = os.environ.get("ADMIN_TOKEN", ADMIN_TOKEN)
    if not token:
        raise HTTPException(503, "admin endpoint disabled: ADMIN_TOKEN is not set")
    if x_admin_token != token:
        raise HTTPException(401, "invalid admin token")
    try:
        version = load_production()
    except Exception as exc:  # noqa: BLE001 - keep serving the old model
        raise HTTPException(500, f"reload failed, still serving version {state.version}: {exc}") from exc
    return {"status": "reloaded", "model_version": version}

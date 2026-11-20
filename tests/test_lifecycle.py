"""Integration tests: train -> register -> promote -> serve -> drift -> retrain -> rollback,
against a throwaway local MLflow (SQLite file), with the real FastAPI app.
The tests run in file order: each one builds on the registry state of the previous one."""

from __future__ import annotations

import mlflow
import pytest
from fastapi.testclient import TestClient

from taxi.config import model_name
from taxi.model import registry
from taxi.model.core import alias_version
from taxi.model.train import train_candidate

GOOD = {"zone_id": 237, "hour_start": "2020-02-14T18:00:00", "lag_1": 150, "lag_2": 140, "lag_3": 120,
        "lag_24": 145, "lag_168": 150, "roll_mean_24": 90.0}


@pytest.fixture(scope="module")
def env(tmp_path_factory, features):
    tmp = tmp_path_factory.mktemp("mlops")
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp)  # local artifacts go to tmp/mlruns
    mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp}/mlflow.db")
    mp.setenv("ADMIN_TOKEN", "test-token")
    mp.setenv("PREDICTION_LOG", str(tmp / "logs" / "predictions.jsonl"))
    mp.setattr("taxi.monitoring.drift.REPORTS_DIR", tmp / "reports")
    mp.setattr("taxi.model.registry.API_URL", "http://127.0.0.1:9")  # no API running: reload just warns
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    path = tmp / "features.parquet"
    features.to_parquet(path, index=False)
    yield {"tmp": tmp, "features_path": path}
    mp.undo()


def test_api_without_model_is_degraded(env):
    from taxi.api.app import app
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=GOOD).status_code == 503


def test_train_logs_lineage_and_registers_staging(env, features):
    res = train_candidate(features, "2019-12-01", "2020-01-31", "2020-02-01", "2020-02-29",
                          data_path=env["features_path"], run_name="test_reference")
    assert res["gate_passed"] and res["version"] == "1"
    assert res["valid"]["wape"] < res["naive"]["wape"]
    run = mlflow.get_run(res["run_id"])
    for tag in ("git_commit", "git_dirty", "data_md5", "trigger", "gate"):
        assert tag in run.data.tags
    assert {"valid_wape", "naive_valid_wape", "valid_mae"} <= set(run.data.metrics)
    assert run.data.params["train_end"] == "2020-01-31"
    assert alias_version(model_name(), "staging").version == "1"


def test_promote_moves_production(env):
    assert registry.promote(reason="first model") == "1"
    assert alias_version(model_name(), "production").version == "1"
    assert alias_version(model_name(), "staging") is None


def test_predict_contract_and_metrics(env):
    from taxi.api.app import app
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200 and h.json()["model_version"] == "1"

        r = client.post("/predict", json=GOOD)
        assert r.status_code == 200
        body = r.json()
        assert body["model_version"] == "1" and body["predicted_pickups"] >= 0
        assert "X-Process-Time-Ms" in r.headers and r.headers["X-Request-ID"] == body["request_id"]

        assert client.post("/predict", json={**GOOD, "lag_1": -3}).status_code == 422
        assert client.post("/predict", json={**GOOD, "zone_id": 999}).status_code == 422
        assert client.post("/predict", json={**GOOD, "fare": 12}).status_code == 422       # unknown field
        assert client.post("/predict", json={**GOOD, "zone_id": 1}).status_code == 422     # zone not served
        assert client.get("/model").json()["train_window"] == ["2019-12-01", "2020-01-31"]

        metrics = client.get("/metrics/").text
        assert 'taxi_predictions_total{model_version="1"}' in metrics
        assert "taxi_request_seconds_bucket" in metrics


def test_live_psi_gauge_rises_on_collapsed_inputs(env, features):
    from taxi.api.app import app
    rows = features[(features.hour_start >= "2020-04-01")].sample(60, random_state=1)
    with TestClient(app) as client:
        for _, r in rows.iterrows():
            payload = {k: float(r[k]) for k in ("lag_1", "lag_2", "lag_3", "lag_24", "lag_168", "roll_mean_24")}
            payload.update(zone_id=int(r.zone_id), hour_start=r.hour_start.isoformat())
            assert client.post("/predict", json=payload).status_code == 200
        metrics = client.get("/metrics/").text
    psi_line = [l for l in metrics.splitlines() if l.startswith('taxi_input_psi{feature="lag_1"}')][0]
    assert float(psi_line.split()[-1]) > 0.25


def test_admin_reload_needs_the_token(env):
    from taxi.api.app import app
    with TestClient(app) as client:
        assert client.post("/admin/reload").status_code == 401
        assert client.post("/admin/reload", headers={"X-Admin-Token": "wrong"}).status_code == 401
        r = client.post("/admin/reload", headers={"X-Admin-Token": "test-token"})
        assert r.status_code == 200 and r.json()["model_version"] == "1"


def test_drift_check_decides_retrain_on_the_collapse(env, features):
    from taxi.monitoring.drift import run_check
    report = run_check(features, "2020-04-01", "2020-04-30", "test-2020-04")
    assert report["decision"] == "RETRAIN"
    assert report["psi"]["lag_1"]["level"] == "act"
    assert (env["tmp"] / "reports" / "drift" / "test-2020-04_v1" / "report.json").exists()


def test_drift_check_flags_a_silent_zone_as_a_pipeline_bug(env, zone_hour):
    from taxi.data.features import build_features
    from taxi.monitoring.drift import run_check
    broken = zone_hour.copy()
    broken.loc[(broken.zone_id == 237) & (broken.hour_start >= "2020-02-10"), "pickups"] = 0
    report = run_check(build_features(broken), "2020-02-01", "2020-02-29", "test-silent")
    assert report["decision"] == "FIX_PIPELINE"
    assert report["quality_issues"][0]["zone_id"] == 237


def test_retrain_is_gated_against_production(env, features):
    res = train_candidate(features, "2020-03-20", "2020-04-23", "2020-04-24", "2020-04-30",
                          data_path=env["features_path"], run_name="test_retrain",
                          compare_production=True, trigger="drift:test")
    assert res["production_version"] == "1"
    assert res["gate_passed"] and res["version"] == "2"
    assert res["valid"]["wape"] <= res["production"]["wape"]


def test_promote_then_rollback(env):
    registry.promote(reason="retrained on the lockdown")
    assert alias_version(model_name(), "production").version == "2"
    assert alias_version(model_name(), "previous").version == "1"

    assert registry.rollback(reason="test rollback") == "1"
    assert alias_version(model_name(), "production").version == "1"
    assert alias_version(model_name(), "previous") is None
    tags = mlflow.MlflowClient().get_model_version(model_name(), "2").tags
    assert tags["rollback_reason"] == "test rollback"

    with pytest.raises(SystemExit):
        registry.rollback()   # nothing left to roll back to without an explicit version

    from taxi.api.app import app
    with TestClient(app) as client:
        assert client.post("/predict", json=GOOD).json()["model_version"] == "1"

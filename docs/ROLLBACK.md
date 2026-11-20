# Rollback runbook

The API serves `models:/nyc-taxi-demand@production`. A rollback moves that alias back to an earlier version and asks the API to reload. Nothing is rebuilt or redeployed, and every version stays in the Registry, so a rollback takes seconds and can itself be undone.

## When to roll back

Roll back when **the last deployment** made things worse:

- the drift check decides `ROLLBACK` (`policy.yaml`, section `rollback`). On fresh labels, one of these holds:
  - production WAPE is more than 10% above `@previous`;
  - a busy zone (500+ pickups in the window) is more than 2x worse;
  - production misses its own validation WAPE by more than 3x while `@previous` stays within 1.5x of its own. That is the signature of leakage or train/serve skew, and it caught the incident's leaky model;
- right after a promotion, the API answers 5xx, `/health` is 503, or `LatencyP95AboveBudget` fires;
- the prediction histogram (`taxi_predicted_pickups`) jumps right after a promotion while the inputs did not move (PSI stable).

Do **not** roll back for drift in the data: an older model is not better on new data. That is a `RETRAIN`. Do not roll back for an upstream bug either: that is a `FIX_PIPELINE`.

## How

```bash
make status                                   # 1. who is @production, who is @previous
make rollback REASON="short reason + link"    # 2. @production -> @previous, the API reloads
# or, to a given version:
make rollback VERSION=2 REASON="..."
make status                                   # 3. check the aliases and the rollback tags
curl -s localhost:8000/model                  # 4. check the version the API actually serves
```

What `make rollback` does (`src/taxi/model/registry.py`):

1. It sets `@production` to the `@previous` version (or `VERSION`).
2. It deletes `@previous`, so a second rollback by mistake cannot move further back silently.
3. It tags the version being removed with `rolled_back_at`, `rolled_back_by` and `rollback_reason`, which `make status` displays.
4. It calls `POST /admin/reload` with `ADMIN_TOKEN`, then prints the version the API now serves (`OK` or `MISMATCH`).

If the API could not be reached in step 4, run `make reload` once it is back. The API keeps serving the old version until then; a failed reload never leaves it without a model.

## Check that it worked

- `make predict`: the `model_version` field is the restored version.
- Prometheus: `taxi_model_version_info{model_version="<restored>"} == 1`, and the rate of `taxi_predictions_total` moves to that label.
- `make check MONTH=<window>`: the decision no longer says `ROLLBACK`.

## After the rollback

1. Write down what happened in the incident log (see `docs/INCIDENT_REPORT.md` for the format).
2. Find the cause, using the lineage of the bad version in MLflow: `git_commit`, `data_md5` (compare with `dvc.lock`), `data_path`, `trigger`, training window.
3. Fix it, retrain (`make retrain`), let the gate decide, and promote again: that is the roll forward.

## Rehearse it

A rollback you have never tried is not a procedure (Reddit, Pi Day 2023). `tests/test_lifecycle.py::test_promote_then_rollback` runs it at every push, and `scripts/run_incident.sh` runs it on the real data.

## Container rollback (code, not model)

If the problem is the API code and not the model, redeploy the previous image tag. Tags are immutable versions, never `latest`:

```bash
API_IMAGE_TAG=1.0.0 docker compose up -d api
```

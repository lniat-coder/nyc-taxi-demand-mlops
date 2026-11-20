# Incident report: the spring the forecast broke

> **Simulated incident on real data.** One real drift (the COVID-19 lockdown in the TLC yellow taxi records) and two injected bugs. It was replayed with `bash scripts/run_incident.sh` on a fresh registry on 2026-11-20; the full console output is in `reports/incident_run.log`. Dates are those of the data, as if the system had been running in 2020. Every number below comes from that run.

| | |
|---|---|
| Service | `nyc-taxi-demand` forecast API, hourly pickups for the 60 busiest yellow taxi zones |
| Impact | v1 over-forecast April 2020 by 52% (324,832 predicted pickups vs 214,003 actual), WAPE 0.629 instead of 0.114, while every request still answered `200 OK` |
| Detection | live PSI (Prometheus), batch drift check, quality check, validation-gap check |
| Reactions | 1 retrain, 1 refused retrain, 1 rollback |
| Versions | v1 (reference) -> v2 (retrained on the lockdown) -> v3 (leaky, rolled back) -> v2 |

## Summary

1. **Real drift, retrain.** In March 2020 the lockdown made taxi demand collapse. v1 had been trained on December and January and kept forecasting pre-COVID volumes. The live PSI jumped above 5, the batch check decided `RETRAIN`, and the candidate v2 passed the gate (WAPE 0.321 vs 0.578 for v1 on the same holdout) and was promoted.
2. **Injected bug A, retrain refused.** In May, zone 237 (Upper East Side South) stopped reporting trips, a simulated broken ingestion. The PSI barely moved. The quality check flagged it and decided `FIX_PIPELINE`, and nothing was retrained on the broken data, even with `--force`.
3. **Injected bug B, rollback.** A feature refactor dated 2020-04-15 forgot a shift, so `lag_1` became the target. A scheduled retrain on that data showed a validation WAPE of 0.001, passed the gate and was auto-promoted as v3. On the clean June labels v3 missed its validation by x355 while v2 stayed at x1.09. The check decided `ROLLBACK`, and `make rollback` restored v2 in 3.7 s.

## Timeline

| Date (data) | What happened | Signal | Evidence |
|---|---|---|---|
| 2020-02 | v1 in production (trained 2019-12-01..2020-01-31), normal traffic | live PSI ≤ 0.012 on all features; batch PSI ≤ 0.004; WAPE 0.114 = validation (ratio 1.00); 6,140,537 pickups forecast for 6,107,196 actual; decision `OK` | `reports/drift/2020-02_v1/report.json` |
| 2020-03-22 | NY "PAUSE" order: demand collapses | — | — |
| 2020-04 | 1,500 April requests replayed on v1 | live PSI 5.59 (`lag_1`) to 7.49 (`roll_mean_24`); `FeatureDriftAct` firing | Prometheus `/alerts` |
| 2020-04-30 | Batch check of April | PSI 5.71 to 7.53 (`act` on all four features); WAPE 0.629 = 5.51x validation; decision `RETRAIN` | `reports/drift/2020-04_v1/report.json`, `evidently.html` |
| 2020-04-30 | Retrain: train 2020-03-20..04-23, holdout 04-24..04-30 | candidate WAPE 0.321 vs v1 0.578 and naive baseline 0.401: gate passed, v2 `@staging` | MLflow run `retrain_2020-04` |
| 2020-04-30 | Human promotion | `@production` 1 -> 2; April replayed again: live PSI 0.013 (`lag_1`), 0.030 (`roll_mean_24`) against v2's own reference (`lag_168` still 0.32: the first training week of v2 carried pre-lockdown lags) | `make status` |
| 2020-05 | **Bug A injected**: zone 237 at 0 for the whole month (744 hours, 10,236 pickups removed) | PSI max 0.155 (`lag_168`, watch): one silent zone is diluted among 60. Quality check: `silent_zone` 237, 744 zero hours, expected 13.1 pickups/hour at the current city volume; decision `FIX_PIPELINE` | `reports/drift/2020-05-features_zone_silent_v2/report.json` |
| 2020-05 | Retrain with `--force` | refused: `--force` never overrides `FIX_PIPELINE` | `reports/incident_run.log` |
| 2020-05-31 | **Bug B injected**: `lag_1` = target on 67,680 rows from 2020-04-15; scheduled retrain (decision was `WATCH`) | candidate v3 WAPE **0.0010** vs v2 0.311 and naive 0.402: gate passed, auto-promoted | `make status`: v3 `data_md5` 96a6fae8 ≠ 662a44de (`dvc.lock`) |
| 2020-06 | Check on clean June labels | v3 WAPE 0.355 = **354.7x** its validation; v2 WAPE 0.351 = 1.09x its own; decision `ROLLBACK` | `reports/drift/2020-06_v3/report.json` |
| 2020-06 | `make rollback` | `@production` 3 -> 2 in 3.7 s, API reloaded and checked, v3 tagged with the reason | `make status` |
| 2020-06 | Check again with v2 | WAPE 0.351 (1.09x validation); PSI `roll_mean_24` 0.693 (June reopening); decision `WATCH` | `reports/drift/2020-06_v2/report.json` |

## Detection: which layer saw what

| Event | Layer 1, infra | Layer 2, inputs | Layer 3, predictions | Layer 4, outcome |
|---|---|---|---|---|
| Lockdown | green: 1,500/1,500 requests in 200 | **PSI > 5**, alert | forecast volume +52% vs actual | **WAPE x5.5** |
| Silent zone | green | PSI ≤ 0.155 (diluted) | — | **quality check** |
| Leaky `lag_1` | green | PSI watch, not specific | volume right (496,642 vs 496,463) | **validation gap x355 vs x1.09** |

The infrastructure layer stayed green in all three events: a wrong model does not crash. Each bug was caught by a different check, which is why the system has all of them.

## What the real data taught us

The first two replays on the real TLC data (`reports/incident_run_1_false_positive.log` and `reports/incident_run_2_no_rollback_signal.log`) did not behave as planned. Both led to a rule change.

1. **False positive in the quality check.** In April 2020, zones 45 (Chinatown, 32 h) and 209 (Seaport, 24 h) really had no taxi for a day: the whole city was down by about 96%. The rule compared each zone to its pre-COVID mean (16.9 and 18.1 pickups per hour), so it flagged a real event as a pipeline bug and blocked the retraining that was needed. **Fix**: the expected level of a zone is now its reference mean multiplied by the city's current volume ratio. Zone 237 in May is still flagged (13.1 expected per hour, 0 observed); zones 45 and 209 in April are not.
2. **The leaky model was not worse than v2.** Having learned "next hour ≈ this hour", v3 behaved like a persistence forecast. As taxis came back in June that forecast was as good as v2 (WAPE 0.355 vs 0.351, worst zone x1.38), so neither "1.10x worse than previous" nor "a zone 2x worse" fired. What gave it away was the gap with its own validation: x355, while v2 kept its promise (x1.09). **Fix**: a third rollback rule: production live WAPE above 3x its validation WAPE **and** the previous model within 1.5x of its own. Real drift degrades both models, so the rule stays silent then; a leak or a train/serve skew only breaks the new one.

Each fix comes with a test (`tests/test_data.py`) and lives in `policy.yaml`, so the thresholds are reviewed like code.

## Root causes

1. **Lockdown**: data and concept drift. The inputs (recent counts) and their relation to the next hour both changed. Nothing was broken: the model was out of date.
2. **Silent zone**: an upstream ingestion bug (simulated). It looks like drift but is not: retraining on it would teach the model that zone 237 has no demand.
3. **Leaky feature**: train/serve skew. Training used a feature computed differently from the one the API receives. The gate compares models on a holdout built by the same broken code, so it cannot see the leak.

## What went well

- The decision tree took a different action for each event: retrain, refuse, roll back.
- The retrained v2 was gated against production on the same holdout and nearly halved the error (0.578 -> 0.321).
- The rollback was one command, took 3.7 s, needed no image rebuild, and was verified against the running API.
- The lineage pointed at the cause: v3's `data_md5` (96a6fae8) does not match the `features.parquet` md5 in `dvc.lock` (662a44de).

## What went badly

- The gate passed a model that was too good to be true: WAPE fell from 0.311 to 0.001 at once.
- `--force` plus `--auto-promote` put that model in production without a human looking at it.
- The silent zone was invisible to PSI. Without the quality check, the next scheduled retrain would have learned it.
- Two of our rules were wrong on real data (see above), which we only found by running on it.
- After the rollback, v2 under-forecasts June by 28% (356,471 vs 496,463) while its WAPE ratio (1.09) stays under the 1.20 retraining threshold: the volume bias is not yet a trigger.

## Actions

| Action | Status |
|---|---|
| Quality checks run before any retraining; `--force` cannot override them | done |
| Silent-zone check scaled by the city's volume | done |
| Validation-gap rollback rule | done |
| `auto_promote: false` by default: a human moves `@production` | done |
| Rollback rehearsed at every push (`tests/test_lifecycle.py`) | done |
| "Too good to be true" gate: hold back a WAPE gain above a threshold until a human reviews it | to do |
| Volume-bias check (layer 4) as a retraining trigger | to do |
| Run the check every night (Airflow) instead of by hand | to do |

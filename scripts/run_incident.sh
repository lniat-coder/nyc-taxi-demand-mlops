#!/usr/bin/env bash
# Replays the whole incident of docs/INCIDENT_REPORT.md, step by step, and keeps every
# report under reports/. Needs: `make data`, `make up`, and an empty registry
# (first run, or `docker compose down -v` to start again from zero).
#
#   bash scripts/run_incident.sh            # pauses between steps (presentation mode)
#   NO_PAUSE=1 bash scripts/run_incident.sh # straight through
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
ZONE=${ZONE:-237}   # Upper East Side South, one of the busiest zones in 2020
export PY
step() { echo; echo "================ $* ================"; [ -n "${NO_PAUSE:-}" ] || read -rp "press Enter "; }

step "0. Reference model: trained Dec 2019-Jan 2020, validated Feb 2020"
make train
make promote REASON="reference model"

step "1. Normal traffic (February 2020): PSI stays under 0.10"
make replay START=2020-02-01 END=2020-02-29 ARGS="--n 1500 --rps 50"
curl -s "${API_URL:-http://localhost:8000}/metrics/" | grep '^taxi_input_psi'
make check MONTH=2020-02

step "2. Lockdown (April 2020): live PSI above 0.25, FeatureDriftAct fires in Prometheus"
make replay START=2020-04-01 END=2020-04-30 ARGS="--n 1500 --rps 50"
curl -s "${API_URL:-http://localhost:8000}/metrics/" | grep '^taxi_input_psi'
echo "-> http://localhost:9090/alerts"

step "3. Retraining trigger on April: RETRAIN, gate, @staging, then a human promotes"
make retrain MONTH=2020-04
make promote REASON="lockdown drift, retrained on the last 35 days of April"
make replay START=2020-04-01 END=2020-04-30 ARGS="--n 1500 --rps 50"
curl -s "${API_URL:-http://localhost:8000}/metrics/" | grep '^taxi_input_psi'

step "4. Injected bug A: zone $ZONE goes silent in May. Quality check: FIX_PIPELINE, no retrain"
make inject ZONE=$ZONE START=2020-05-01 END=2020-05-31
make retrain MONTH=2020-05 FEATURES=data/injected/features_zone_silent.parquet ARGS="--force"

step "5. Injected bug B: lag_1 off-by-one (leaks the target) from 2020-04-15. Scheduled retrain passes the gate"
$PY scripts/inject_drift.py --mode leaky_feature --start 2020-04-15 --end 2020-05-31
make retrain MONTH=2020-05 FEATURES=data/injected/features_leaky_feature.parquet ARGS="--force --auto-promote"
make status

step "6. Fresh labels (June 2020, clean data): production worse than previous -> ROLLBACK"
make check MONTH=2020-06

step "7. Rollback: @production back to the previous version, API reloaded"
time make rollback REASON="lag_1 off-by-one leak in the training data (drift check 2020-06)"
make status
make check MONTH=2020-06

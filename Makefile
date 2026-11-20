# Every command of the project. `make help` lists them.
# Windows without make: run the command shown after each target in a terminal.

-include .env
export

PY ?= .venv/bin/python
MONTH ?= 2020-04
REASON ?=

.PHONY: help setup data up down logs train promote rollback reload status check retrain inject replay predict test clean

help:  ## list the commands
	@grep -E '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

setup:  ## virtualenv + dependencies + .env
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt -e .
	@test -f .env || (cp .env.example .env && echo "created .env: change ADMIN_TOKEN")

data:  ## download the TLC files and build the features (DVC pipeline)
	PATH="$(CURDIR)/.venv/bin:$$PATH" dvc repro

up:  ## start MLflow, the API and Prometheus
	docker compose up -d --build

down:  ## stop the stack (the MLflow volume is kept)
	docker compose down

logs:  ## follow the API logs
	docker compose logs -f api

train:  ## train the reference model, register it as @staging if it beats the naive baseline
	$(PY) -m taxi.model.train

promote:  ## move @production to @staging (or VERSION=N) and reload the API
	$(PY) -m taxi.model.registry promote $(if $(VERSION),--version $(VERSION)) --reason "$(REASON)"

rollback:  ## move @production back to @previous (or VERSION=N) and reload the API
	$(PY) -m taxi.model.registry rollback $(if $(VERSION),--to-version $(VERSION)) --reason "$(REASON)"

reload:  ## ask the API to reload @production
	curl -s -X POST -H "X-Admin-Token: $(ADMIN_TOKEN)" $(API_URL)/admin/reload; echo

status:  ## registry: versions, aliases, data hash, git commit
	$(PY) -m taxi.model.registry status

check:  ## drift check of MONTH (default 2020-04), FEATURES=path for injected data
	$(PY) -m taxi.monitoring.drift --month $(MONTH) $(if $(FEATURES),--features $(FEATURES) --label $(MONTH)-$(notdir $(basename $(FEATURES))))

retrain:  ## drift check of MONTH, then retrain only if the decision is RETRAIN
	$(PY) -m taxi.pipelines.check_and_retrain --month $(MONTH) $(if $(FEATURES),--features $(FEATURES) --label $(MONTH)-$(notdir $(basename $(FEATURES)))) $(ARGS)

inject:  ## inject a silent zone (ZONE, START, END) into a copy of the data
	$(PY) scripts/inject_drift.py --zone $(or $(ZONE),237) --start $(or $(START),2020-05-01) --end $(or $(END),2020-05-31)

replay:  ## replay START..END as live traffic on the API
	$(PY) scripts/replay_traffic.py --start $(or $(START),2020-02-01) --end $(or $(END),2020-02-29) $(ARGS)

predict:  ## one example request
	curl -s -X POST $(API_URL)/predict -H "Content-Type: application/json" \
	  -d '{"zone_id": 237, "hour_start": "2020-02-14T18:00:00", "lag_1": 310, "lag_2": 295, "lag_3": 260, "lag_24": 280, "lag_168": 301, "roll_mean_24": 190.5}'; echo

test:  ## unit and integration tests (no network, no Docker needed)
	$(PY) -m pytest -q

clean:  ## remove local reports and logs (not the data, not MLflow)
	rm -rf reports/drift logs/*.jsonl

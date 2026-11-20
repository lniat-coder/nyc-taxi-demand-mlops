"""Retraining trigger: run the drift check, and retrain only if the decision says so.

    check (taxi.monitoring.drift) --RETRAIN--> train candidate on the last days of the window
                                               --> gate vs production on the holdout days
                                               --> register, alias @staging
                                               --> a human promotes (or --auto-promote)

A failed gate costs a training run; the old model stays in production.
On ROLLBACK or FIX_PIPELINE nothing is retrained and the command to run is printed.

    python -m taxi.pipelines.check_and_retrain --month 2020-04
    python -m taxi.pipelines.check_and_retrain --month 2020-05 --force   # retrain anyway (scheduled floor)

--force retrains even when the inputs did not drift (the "scheduled floor" of the course's
hybrid policy), and is logged as trigger=scheduled. It never overrides FIX_PIPELINE or
ROLLBACK: suspicious data and a bad production model are handled by a human.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from taxi.model.registry import promote, reload_api
from taxi.model.train import train_candidate
from taxi.monitoring.drift import resolve_window, run_check, window_args
from taxi.config import policy

log = logging.getLogger(__name__)


def retrain_windows(end: str, train_days: int, holdout_days: int) -> tuple[str, str, str, str]:
    """Train on the `train_days` before the holdout, hold out the last `holdout_days` of the window."""
    end_ts = pd.Timestamp(end)
    valid_start = end_ts - pd.Timedelta(days=holdout_days - 1)
    train_end = valid_start - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=train_days - 1)
    return tuple(str(t.date()) for t in (train_start, train_end, valid_start, end_ts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    window_args(ap)
    ap.add_argument("--force", action="store_true", help="retrain whatever the decision (logged as trigger=forced)")
    ap.add_argument("--auto-promote", action="store_true", help="promote if the gate passes (default: policy.yaml)")
    a = ap.parse_args()

    start, end, label = resolve_window(a)
    features = pd.read_parquet(a.features)
    report = run_check(features, start, end, label)
    decision = report["decision"]

    if decision == "ROLLBACK":
        print('Not retraining. Next: make rollback REASON="..."  (see docs/ROLLBACK.md)')
        return
    if decision == "FIX_PIPELINE":
        print("Not retraining on suspicious data (even with --force). Fix the upstream pipeline, then re-run.")
        return
    if decision != "RETRAIN" and not a.force:
        print(f"No retraining ({decision}).")
        return

    r = policy()["retrain"]
    tr_s, tr_e, va_s, va_e = retrain_windows(end, r["train_days"], r["holdout_days"])
    trigger = f"drift:{label}" if decision == "RETRAIN" else "scheduled"
    if trigger == "scheduled":
        print(f"Scheduled retrain (decision was {decision}).")
    res = train_candidate(features, tr_s, tr_e, va_s, va_e, data_path=a.features.resolve(),
                          run_name=f"retrain_{label}", compare_production=True, trigger=trigger,
                          extra_tags={"drift_decision": decision, "drift_window": f"{start}..{end}"})
    if not res["gate_passed"]:
        print("Gate failed: candidate not registered, production unchanged.")
        return
    print(f"Candidate version {res['version']} registered as @staging "
          f"(WAPE {res['valid']['wape']:.3f} vs production {res['production']['wape']:.3f} on {va_s}..{va_e}).")
    if a.auto_promote or r["auto_promote"]:
        promote(res["version"], reason=f"auto-promoted after {trigger}")
        reload_api(res["version"])
    else:
        print('Next: make promote REASON="..."')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

"""Batch drift check of a time window: from an alarm to a decision.

For the window, against the @production model and the data it was trained on:
  1. Is it a bug?        data quality checks (silent zone, empty hours, duplicates)
  2. Did the model move? production vs @previous on fresh labels, and each model's
                         live WAPE vs its own validation WAPE (rollback signal)
  3. Did the data move?  PSI per feature (+ an Evidently HTML report)
  4. Does it hurt?       production WAPE on the window / its validation WAPE
  5. Enough new data?    number of days in the window

Decisions: OK, WATCH, FIX_PIPELINE, WAIT_FOR_DATA, RETRAIN, ROLLBACK.
Everything is written to reports/drift/<label>_v<model version>/ and logged to MLflow.

    python -m taxi.monitoring.drift --month 2020-04
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import mlflow
import pandas as pd

from taxi.config import FEATURES_PATH, MONITORING_EXPERIMENT, REPORTS_DIR, model_name, params, policy
from taxi.data.features import time_slice
from taxi.data.quality import check_window
from taxi.model.core import alias_version, init_mlflow, load_version, regression_metrics, to_model_input
from taxi.monitoring.psi import level, psi

log = logging.getLogger(__name__)


def decide(issues: list, psi_max: float, wape_ratio: float, n_days: int, rollback_signal: bool,
           pol: dict) -> tuple[str, str]:
    """The decision tree of the course (session 5), in order."""
    if issues:
        return "FIX_PIPELINE", "data quality issue: a broken pipeline looks like drift, do not retrain on it"
    if rollback_signal:
        return "ROLLBACK", ("on fresh labels the production model does worse than the previous one, "
                            "or breaks its validation promise while the previous one keeps it")
    if psi_max <= pol["drift"]["psi_watch"]:
        return "OK", "inputs look like the training data"
    if psi_max <= pol["drift"]["psi_act"]:
        return "WATCH", "moderate drift, keep watching"
    if wape_ratio < pol["performance"]["hurt_ratio"]:
        return "WATCH", "strong drift but the error has not degraded enough to retrain"
    if n_days < pol["retrain"]["min_days"]:
        return "WAIT_FOR_DATA", "drift hurts but the window is too short to train on"
    return "RETRAIN", "inputs drifted and the error degraded"


def zone_wape(df: pd.DataFrame, pred_col: str, target: str) -> pd.Series:
    g = df.assign(err=(df[target] - df[pred_col]).abs()).groupby("zone_id")
    return g["err"].sum() / g[target].sum().clip(lower=1)


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], path: Path) -> bool:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        log.warning("evidently not installed: skipping the HTML report")
        return False
    ref = reference[columns].sample(min(len(reference), 5000), random_state=0)
    cur = current[columns].sample(min(len(current), 5000), random_state=0)
    Report([DataDriftPreset()]).run(cur, ref).save_html(str(path))
    return True


def run_check(features: pd.DataFrame, start: str, end: str, label: str) -> dict:
    init_mlflow()
    pol, target, name = policy(), params()["features"]["target"], model_name()
    prod = alias_version(name, "production")
    if prod is None:
        raise SystemExit("No @production model to monitor.")
    mv, model, profile = load_version(name, prod.version)
    columns = profile["feature_columns"]

    window = time_slice(features, start, end).copy()
    if window.empty:
        raise SystemExit(f"No data between {start} and {end}.")
    n_days = window.hour_start.dt.normalize().nunique()

    # 1. Is it a bug?
    q = pol["quality"]
    issues = check_window(window[["zone_id", "hour_start", target]],
                          {int(z): m for z, m in profile["zone_means"].items()},
                          q["silent_zone_hours"], q["silent_zone_min_ref_mean"], q["max_empty_hours"])

    # 2. Performance of production on fresh labels, and of the previous version
    window["pred_production"] = model.predict(to_model_input(window, columns)).clip(min=0)
    perf = regression_metrics(window[target], window["pred_production"])
    wape_ratio = perf["wape"] / profile["valid_wape"]
    zones_prod = zone_wape(window, "pred_production", target)

    rb = pol["rollback"]
    previous = alias_version(name, "previous")
    prev_perf, rollback_detail = None, {}
    if previous is not None:
        _, prev_model, prev_profile = load_version(name, previous.version)
        window["pred_previous"] = prev_model.predict(to_model_input(window, columns)).clip(min=0)
        prev_perf = regression_metrics(window[target], window["pred_previous"])
        volume = window.groupby("zone_id")[target].sum()
        busy = volume[volume >= rb["zone_min_pickups"]].index
        ratio_by_zone = (zones_prod / zone_wape(window, "pred_previous", target).clip(lower=1e-6)).loc[busy]
        worst_zone = int(ratio_by_zone.idxmax()) if len(ratio_by_zone) else None
        rollback_detail = {
            "previous_version": previous.version,
            "wape_ratio_vs_previous": perf["wape"] / max(prev_perf["wape"], 1e-9),
            "worst_zone": worst_zone,
            "worst_zone_wape_ratio": float(ratio_by_zone.max()) if len(ratio_by_zone) else None,
            # live WAPE / validation WAPE, for each model: does it keep the promise of its validation?
            "production_validation_gap": wape_ratio,
            "previous_validation_gap": prev_perf["wape"] / prev_profile["valid_wape"],
        }
    rollback_signal = bool(rollback_detail) and (
        rollback_detail["wape_ratio_vs_previous"] > rb["max_wape_ratio"]
        or (rollback_detail["worst_zone_wape_ratio"] or 0) > rb["max_zone_wape_ratio"]
        # The new model breaks its validation promise while the previous one keeps its own:
        # the problem is the new model (leakage, train/serve skew), not the data.
        or (rollback_detail["production_validation_gap"] > rb["max_validation_gap"]
            and rollback_detail["previous_validation_gap"] < rb["previous_ok_gap"]))

    # 3. Did the inputs move?
    d = pol["drift"]
    psi_values = {f: psi(profile["psi"][f], window[f].to_numpy()) for f in profile["psi"]}
    psi_max = max(psi_values.values())

    decision, reason = decide(issues, psi_max, wape_ratio, n_days, rollback_signal, pol)

    report = {
        "label": label, "window": [start, end], "n_rows": len(window), "n_days": int(n_days),
        "model_version": mv.version, "reference_window": profile["train_window"],
        "decision": decision, "reason": reason, "quality_issues": issues,
        "psi": {f: {"value": round(v, 4), "level": level(v, d["psi_watch"], d["psi_act"])}
                for f, v in psi_values.items()},
        "performance": {**{k: round(v, 4) for k, v in perf.items()},
                        "reference_wape": round(profile["valid_wape"], 4), "wape_ratio": round(wape_ratio, 3)},
        "previous": ({**{k: round(v, 4) for k, v in prev_perf.items()}, **rollback_detail}
                     if prev_perf else None),
        "worst_zones": {str(z): round(float(v), 3) for z, v in zones_prod.nlargest(5).items()},
        "volume": {"actual_pickups": int(window[target].sum()),
                   "predicted_pickups": int(window["pred_production"].sum())},
    }

    out = REPORTS_DIR / "drift" / f"{label}_v{mv.version}"  # one folder per window and model
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    reference = time_slice(features, *profile["train_window"])
    has_html = evidently_report(reference, window, list(profile["psi"]), out / "evidently.html")

    mlflow.set_experiment(MONITORING_EXPERIMENT)
    with mlflow.start_run(run_name=f"drift_{label}"):
        mlflow.set_tags({"decision": decision, "model_version": mv.version, "window": f"{start}..{end}"})
        mlflow.log_metrics({**{f"psi_{f}": v for f, v in psi_values.items()},
                            "wape": perf["wape"], "wape_ratio": wape_ratio, "n_quality_issues": len(issues)})
        mlflow.log_artifact(str(out / "report.json"))
        if has_html:
            mlflow.log_artifact(str(out / "evidently.html"))
    print_summary(report, out)
    return report


def print_summary(r: dict, out: Path) -> None:
    print(f"\n=== Drift check {r['label']} ({r['window'][0]} .. {r['window'][1]}), "
          f"model v{r['model_version']} trained on {r['reference_window'][0]} .. {r['reference_window'][1]}")
    for f, v in r["psi"].items():
        print(f"  PSI {f:<14} {v['value']:>7.3f}  {v['level']}")
    p = r["performance"]
    print(f"  WAPE {p['wape']:.3f} (validation {p['reference_wape']:.3f}, ratio {p['wape_ratio']:.2f})")
    print(f"  pickups: actual {r['volume']['actual_pickups']:,}, predicted {r['volume']['predicted_pickups']:,}")
    if r["previous"]:
        pv = r["previous"]
        print(f"  vs previous v{pv['previous_version']}: WAPE {pv['wape']:.3f}, ratio {pv['wape_ratio_vs_previous']:.2f}, "
              f"worst zone {pv['worst_zone']} x{(pv['worst_zone_wape_ratio'] or 0):.2f}")
        print(f"  live/validation WAPE: production x{pv['production_validation_gap']:.2f}, "
              f"previous x{pv['previous_validation_gap']:.2f}")
    for issue in r["quality_issues"]:
        print(f"  QUALITY ISSUE: {issue}")
    print(f"  DECISION: {r['decision']} ({r['reason']})")
    print(f"  report: {out}/report.json\n")


def window_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--month", help="YYYY-MM, shortcut for --start/--end")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--features", type=Path, default=FEATURES_PATH)
    ap.add_argument("--label", help="report folder name (default: the month or the dates)")


def resolve_window(a) -> tuple[str, str, str]:
    if a.month:
        start = pd.Timestamp(f"{a.month}-01")
        end = start + pd.offsets.MonthEnd(0)
        return str(start.date()), str(end.date()), a.label or a.month
    if not (a.start and a.end):
        raise SystemExit("give --month or --start and --end")
    return a.start, a.end, a.label or f"{a.start}_{a.end}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    window_args(ap)
    a = ap.parse_args()
    start, end, label = resolve_window(a)
    run_check(pd.read_parquet(a.features), start, end, label)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

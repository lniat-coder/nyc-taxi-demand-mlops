"""Train a candidate, log everything to MLflow, and register it if it passes the gate.

Every run records what is needed to rebuild it: the parameters, the metrics,
the git commit (and whether the tree was dirty), the md5 of the features file
(the same md5 DVC writes in dvc.lock), and the time windows used.

Usage:
    python -m taxi.model.train                       # reference model (periods from params.yaml)
    python -m taxi.model.train --compare-production  # also gate against the production model
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import warnings
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from taxi.config import FEATURES_PATH, ROOT, TRAINING_EXPERIMENT, feature_columns, model_name, params, policy
from taxi.data.features import time_slice
from taxi.model.core import (PROFILE_ARTIFACT, alias_version, init_mlflow, load_version,
                             regression_metrics, to_model_input)
from taxi.monitoring.psi import reference_profile

log = logging.getLogger(__name__)
# zone_id is an int64 on purpose (category) and never missing: MLflow's hint does not apply.
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")

# MLflow 3 saves scikit-learn models with skops, which refuses to load types it does not
# know. We list the ones our pipeline needs, after review, instead of falling back to pickle.
SKOPS_TRUSTED_TYPES = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]


def git_info() -> dict[str, str]:
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"git_commit": commit or "unknown", "git_dirty": str(bool(status)) if commit else "unknown"}


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pipeline(cfg: dict) -> Pipeline:
    # zone_id is the first column: encoded to 0..K-1, unseen zones become -1 (= missing for the model)
    encode = ColumnTransformer(
        [("zone", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["zone_id"])],
        remainder="passthrough", verbose_feature_names_out=False)
    model = HistGradientBoostingRegressor(
        loss=cfg["loss"], learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"],
        max_leaf_nodes=cfg["max_leaf_nodes"], min_samples_leaf=cfg["min_samples_leaf"],
        random_state=cfg["random_state"], categorical_features=[0])
    return Pipeline([("encode", encode), ("model", model)])


def train_candidate(features: pd.DataFrame, train_start: str, train_end: str, valid_start: str,
                    valid_end: str, *, data_path: Path, run_name: str, register: bool = True,
                    compare_production: bool = False, trigger: str = "manual",
                    extra_tags: dict | None = None) -> dict:
    init_mlflow()
    p, pol = params(), policy()
    columns, target = feature_columns(), p["features"]["target"]
    train, valid = time_slice(features, train_start, train_end), time_slice(features, valid_start, valid_end)
    if train.empty or valid.empty:
        raise ValueError("empty train or validation window")
    if train.hour_start.max() >= valid.hour_start.min():
        raise ValueError("validation must start after training ends (split by time)")

    pipe = build_pipeline(p["model"])
    X_train, X_valid = to_model_input(train, columns), to_model_input(valid, columns)
    pipe.fit(X_train, train[target])

    m_valid = regression_metrics(valid[target], pipe.predict(X_valid))
    m_train = regression_metrics(train[target], pipe.predict(X_train))
    m_naive = regression_metrics(valid[target], valid["lag_168"])  # "same hour last week"

    # Gate: beat the naive baseline, and beat production on the same holdout by min_gain.
    prod = alias_version(model_name(), "production") if compare_production else None
    m_prod = None
    if prod is not None:
        _, prod_model, _ = load_version(model_name(), prod.version)
        m_prod = regression_metrics(valid[target], prod_model.predict(X_valid))
    min_gain = pol["retrain"]["min_gain"]
    gate_baseline = m_valid["wape"] < m_naive["wape"]
    gate_production = m_prod is None or m_valid["wape"] <= m_prod["wape"] * (1 - min_gain)
    passed = gate_baseline and gate_production

    profile = {
        "feature_columns": columns,
        "psi": reference_profile(train, pol["drift"]["features"]),
        "zone_means": {str(z): float(v) for z, v in train.groupby("zone_id")[target].mean().items()},
        "zones": sorted(int(z) for z in train.zone_id.unique()),
        "valid_wape": m_valid["wape"],
        "train_window": [train_start, train_end],
        "valid_window": [valid_start, valid_end],
    }
    tags = {**git_info(), "data_md5": file_md5(data_path), "data_path": str(data_path.relative_to(ROOT))
            if data_path.is_relative_to(ROOT) else str(data_path),
            "trigger": trigger, "gate": "passed" if passed else "failed", **(extra_tags or {})}

    mlflow.set_experiment(TRAINING_EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params({**{f"model_{k}": v for k, v in p["model"].items()},
                           "train_start": train_start, "train_end": train_end,
                           "valid_start": valid_start, "valid_end": valid_end,
                           "n_train": len(train), "n_valid": len(valid), "n_zones": train.zone_id.nunique(),
                           "features": ",".join(columns), "min_gain": min_gain})
        mlflow.log_metrics({**{f"valid_{k}": v for k, v in m_valid.items()},
                            **{f"train_{k}": v for k, v in m_train.items()},
                            **{f"naive_valid_{k}": v for k, v in m_naive.items()},
                            **({f"production_valid_{k}": v for k, v in m_prod.items()} if m_prod else {})})
        mlflow.log_dict(profile, PROFILE_ARTIFACT)
        info = mlflow.sklearn.log_model(
            pipe, name="model", signature=infer_signature(X_valid, pipe.predict(X_valid)),
            input_example=X_valid.head(3), skops_trusted_types=SKOPS_TRUSTED_TYPES)

        version = None
        if register and passed:
            mv = mlflow.register_model(info.model_uri, model_name())
            version = str(mv.version)
            client = mlflow.MlflowClient()
            for k in ("git_commit", "data_md5", "trigger"):
                client.set_model_version_tag(model_name(), version, k, tags[k])
            client.set_model_version_tag(model_name(), version, "train_window", f"{train_start}..{train_end}")
            client.set_model_version_tag(model_name(), version, "valid_wape", f"{m_valid['wape']:.4f}")
            client.set_registered_model_alias(model_name(), "staging", version)

    result = {"run_id": run.info.run_id, "version": version, "gate_passed": passed,
              "valid": m_valid, "naive": m_naive, "production": m_prod,
              "production_version": prod.version if prod else None}
    log.info("candidate: WAPE %.4f | naive %.4f | production %s | gate %s | version %s",
             m_valid["wape"], m_naive["wape"], f"{m_prod['wape']:.4f}" if m_prod else "-",
             "passed" if passed else "FAILED", version or "not registered")
    return result


def main() -> None:
    s = params()["split"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=FEATURES_PATH)
    ap.add_argument("--train-start", default=s["train_start"])
    ap.add_argument("--train-end", default=s["train_end"])
    ap.add_argument("--valid-start", default=s["valid_start"])
    ap.add_argument("--valid-end", default=s["valid_end"])
    ap.add_argument("--compare-production", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()

    features = pd.read_parquet(a.features)
    res = train_candidate(features, a.train_start, a.train_end, a.valid_start, a.valid_end,
                          data_path=a.features.resolve(), run_name=f"train_{a.train_start}_{a.train_end}",
                          register=not a.no_register, compare_production=a.compare_production)
    if res["version"]:
        print(f"Registered {model_name()} version {res['version']} with alias @staging.")
        print("Promote it with: make promote")
    elif not res["gate_passed"]:
        raise SystemExit("Gate failed: the candidate was logged but not registered.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

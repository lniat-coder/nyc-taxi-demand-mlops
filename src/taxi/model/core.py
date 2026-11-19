"""Pieces shared by training, monitoring and the API: model inputs, metrics, loading."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

PROFILE_ARTIFACT = "monitoring/reference_profile.json"
CATEGORICAL = ["zone_id"]


@dataclass(frozen=True)
class VersionRef:
    """A registered model version. MLflow returns int or str versions depending on the
    store, so the version is always normalised to a string here."""
    version: str
    run_id: str
    tags: dict = field(default_factory=dict)

    @classmethod
    def of(cls, mv) -> "VersionRef":
        return cls(str(mv.version), mv.run_id, dict(mv.tags or {}))


def init_mlflow() -> None:
    """Point the client at the tracking server (MLFLOW_TRACKING_URI, from .env)."""
    import os

    from taxi.config import MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))


def to_model_input(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Same column order and dtypes everywhere: int64 for categories, float64 otherwise.

    The MLflow signature is inferred from this frame at training time, so the
    API and the batch jobs must build their inputs with this function too.
    """
    out = df[columns].copy()
    for c in columns:
        out[c] = out[c].astype("int64") if c in CATEGORICAL else out[c].astype("float64")
    return out


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        "wape": float(err.sum() / max(y_true.sum(), 1.0)),
    }


def load_version(name: str, version: str | int):
    """Load a registered model version and the reference profile logged with it."""
    mv = VersionRef.of(mlflow.MlflowClient().get_model_version(name, str(version)))
    model = mlflow.pyfunc.load_model(f"models:/{name}/{mv.version}")
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path=PROFILE_ARTIFACT, dst_path=tmp)
        profile = json.loads(Path(path).read_text())
    return mv, model, profile


def alias_version(name: str, alias: str) -> VersionRef | None:
    """The version behind an alias, or None if the alias is not set."""
    try:
        return VersionRef.of(mlflow.MlflowClient().get_model_version_by_alias(name, alias))
    except mlflow.exceptions.MlflowException:
        return None

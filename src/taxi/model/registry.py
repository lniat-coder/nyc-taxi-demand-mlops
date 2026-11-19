"""Promote and roll back models by moving MLflow Registry aliases.

The API always serves `models:/nyc-taxi-demand@production`. Rolling back is
moving that alias back to the previous version and asking the API to reload:
no rebuild, no redeploy, one command. See docs/ROLLBACK.md.

    python -m taxi.model.registry status
    python -m taxi.model.registry promote [--version N] --reason "..."
    python -m taxi.model.registry rollback [--to-version N] --reason "..."
"""

from __future__ import annotations

import argparse
import getpass
from datetime import datetime, timezone

import mlflow
import requests

from taxi.config import ADMIN_TOKEN, API_URL, model_name
from taxi.model.core import alias_version, init_mlflow

PRODUCTION, PREVIOUS, STAGING = "production", "previous", "staging"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _who() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no user in some containers
        return "unknown"


def promote(version: str | None = None, reason: str = "") -> str:
    """Move @production to `version` (default: the @staging version). The old one becomes @previous."""
    client, name = mlflow.MlflowClient(), model_name()
    if version is None:
        staging = alias_version(name, STAGING)
        if staging is None:
            raise SystemExit("No @staging version to promote. Train one first (make train).")
        version = staging.version
    version = str(version)
    client.get_model_version(name, version)  # fails loudly if it does not exist

    current = alias_version(name, PRODUCTION)
    if current is not None and current.version == version:
        print(f"Version {version} is already @production.")
        return version
    if current is not None:
        client.set_registered_model_alias(name, PREVIOUS, current.version)
    client.set_registered_model_alias(name, PRODUCTION, version)
    staging = alias_version(name, STAGING)
    if staging is not None and staging.version == version:
        client.delete_registered_model_alias(name, STAGING)
    client.set_model_version_tag(name, version, "promoted_at", _now())
    client.set_model_version_tag(name, version, "promoted_by", _who())
    client.set_model_version_tag(name, version, "promotion_reason", reason or "-")
    print(f"@production: {current.version if current else 'none'} -> {version}")
    return version


def rollback(to_version: str | None = None, reason: str = "") -> str:
    """Move @production back to @previous (or to an explicit version)."""
    client, name = mlflow.MlflowClient(), model_name()
    current = alias_version(name, PRODUCTION)
    if to_version is None:
        previous = alias_version(name, PREVIOUS)
        if previous is None:
            raise SystemExit("No @previous version: pass --to-version N explicitly.")
        to_version = previous.version
    to_version = str(to_version)
    client.get_model_version(name, to_version)
    client.set_registered_model_alias(name, PRODUCTION, to_version)
    if alias_version(name, PREVIOUS) is not None:
        client.delete_registered_model_alias(name, PREVIOUS)
    if current is not None:
        client.set_model_version_tag(name, current.version, "rolled_back_at", _now())
        client.set_model_version_tag(name, current.version, "rolled_back_by", _who())
        client.set_model_version_tag(name, current.version, "rollback_reason", reason or "-")
    print(f"ROLLBACK @production: {current.version if current else 'none'} -> {to_version}")
    return to_version


def reload_api(expected_version: str | None = None) -> None:
    """Ask the running API to load the new @production version, then check what it serves."""
    try:
        r = requests.post(f"{API_URL}/admin/reload", headers={"X-Admin-Token": ADMIN_TOKEN}, timeout=60)
        r.raise_for_status()
        served = str(r.json().get("model_version"))
    except requests.RequestException as exc:
        print(f"WARNING: API not reloaded ({exc}). Run `make reload` once the API is up.")
        return
    status = "OK" if expected_version in (None, served) else f"MISMATCH, expected {expected_version}"
    print(f"API at {API_URL} now serves version {served}: {status}")


def status() -> None:
    client, name = mlflow.MlflowClient(), model_name()
    try:
        rm = client.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        print(f"No registered model '{name}' yet.")
        return
    by_version: dict[str, list[str]] = {}
    for alias, v in (rm.aliases or {}).items():
        by_version.setdefault(str(v), []).append("@" + alias)
    print(f"{'version':>7}  {'aliases':<24} {'valid_wape':>10}  {'train_window':<24} {'git':<9} {'data_md5':<10} notes")
    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda m: int(m.version)):
        t = mv.tags or {}
        notes = "rolled back: " + t["rollback_reason"] if "rolled_back_at" in t else t.get("trigger", "")
        print(f"{mv.version:>7}  {' '.join(by_version.get(str(mv.version), [])):<24} "
              f"{t.get('valid_wape', '-'):>10}  {t.get('train_window', '-'):<24} "
              f"{t.get('git_commit', '-')[:8]:<9} {t.get('data_md5', '-')[:8]:<10} {notes}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("promote")
    p.add_argument("--version")
    p.add_argument("--reason", default="")
    r = sub.add_parser("rollback")
    r.add_argument("--to-version")
    r.add_argument("--reason", default="")
    for s in (p, r):
        s.add_argument("--no-reload", action="store_true", help="do not call the API /admin/reload")
    a = ap.parse_args()

    init_mlflow()
    if a.cmd == "status":
        status()
        return
    version = promote(a.version, a.reason) if a.cmd == "promote" else rollback(a.to_version, a.reason)
    if not a.no_reload:
        reload_api(version)


if __name__ == "__main__":
    main()

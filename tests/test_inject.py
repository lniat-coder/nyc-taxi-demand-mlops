import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("inject_drift", Path(__file__).parents[1] / "scripts" / "inject_drift.py")
inject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inject)


def test_zone_silent_only_touches_the_zone_and_dates(zone_hour):
    out = inject.zone_silent(zone_hour, 237, "2020-05-01", "2020-05-31")
    hit = (out.zone_id == 237) & (out.hour_start >= "2020-05-01") & (out.hour_start < "2020-06-01")
    assert (out.loc[hit, "pickups"] == 0).all()
    assert out.loc[~hit, "pickups"].equals(zone_hour.loc[~hit, "pickups"])


def test_leaky_feature_puts_the_target_in_lag_1(features):
    out = inject.leaky_feature(features, "2020-04-15", "2020-05-31")
    hit = (out.hour_start >= "2020-04-15") & (out.hour_start < "2020-06-01")
    assert (out.loc[hit, "lag_1"] == out.loc[hit, "pickups"]).all()
    assert out.loc[~hit, "lag_1"].equals(features.loc[~hit, "lag_1"])

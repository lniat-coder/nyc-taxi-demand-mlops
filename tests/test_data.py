import pandas as pd

from taxi.data.aggregate import aggregate_month, complete_grid, select_zones
from taxi.data.features import calendar_features, time_slice
from taxi.data.quality import check_window, longest_zero_run


def test_aggregate_drops_trips_outside_the_month(tmp_path):
    trips = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2020-01-01 00:10", "2020-01-01 00:50", "2020-01-31 23:59",
                                                "2008-12-31 23:00", "2020-02-01 00:00"]),
        "PULocationID": [237, 237, 132, 237, 237],
        "fare_amount": [10.0] * 5})
    path = tmp_path / "yellow_tripdata_2020-01.parquet"
    trips.to_parquet(path)
    counts = aggregate_month(path, "2020-01")
    assert counts.pickups.sum() == 3
    assert counts.set_index(["zone_id", "hour_start"]).loc[(237, pd.Timestamp("2020-01-01 00:00")), "pickups"] == 2


def test_complete_grid_fills_missing_hours_with_zero():
    counts = pd.DataFrame({"zone_id": [237], "hour_start": [pd.Timestamp("2020-01-01 05:00")], "pickups": [7]})
    grid = complete_grid(counts, [132, 237], "2020-01", "2020-01")
    assert len(grid) == 2 * 31 * 24
    assert grid.pickups.sum() == 7
    assert (grid[grid.zone_id == 132].pickups == 0).all()


def test_select_zones_uses_training_period_only():
    counts = pd.DataFrame({"zone_id": [1, 2, 3, 264], "pickups": [100, 50, 1000, 5000],
                           "hour_start": pd.to_datetime(["2020-01-05", "2020-01-05", "2020-03-01", "2020-01-05"])})
    assert select_zones(counts, "2020-01-01", "2020-01-31", top_k=2, excluded=[264]) == [1, 2]


def test_lags_only_look_backwards(features, zone_hour):
    z = zone_hour[zone_hour.zone_id == 237].set_index("hour_start")["pickups"]
    row = features[(features.zone_id == 237)].iloc[100]
    t = row.hour_start
    assert row.pickups == z[t]
    assert row.lag_1 == z[t - pd.Timedelta(hours=1)]
    assert row.lag_168 == z[t - pd.Timedelta(hours=168)]
    assert abs(row.roll_mean_24 - z[t - pd.Timedelta(hours=24): t - pd.Timedelta(hours=1)].mean()) < 1e-9
    assert not features.isna().any().any()


def test_time_slice_is_inclusive_and_never_overlaps(features):
    train = time_slice(features, "2019-12-01", "2020-01-31")
    valid = time_slice(features, "2020-02-01", "2020-02-29")
    assert train.hour_start.max() == pd.Timestamp("2020-01-31 23:00")
    assert valid.hour_start.min() == pd.Timestamp("2020-02-01 00:00")
    assert train.hour_start.max() < valid.hour_start.min()


def test_calendar_features():
    cal = calendar_features(pd.Series(pd.to_datetime(["2020-01-01 08:00", "2020-01-04 18:00"])))
    assert cal.is_holiday.tolist() == [1, 0]
    assert cal.is_weekend.tolist() == [0, 1]
    assert cal.hour.tolist() == [8, 18]


def test_longest_zero_run():
    assert longest_zero_run(pd.Series([1, 0, 0, 3, 0, 0, 0, 2])) == 3
    assert longest_zero_run(pd.Series([1, 2])) == 0


def test_quality_flags_a_silent_busy_zone_only(zone_hour):
    window = zone_hour[(zone_hour.hour_start >= "2020-02-01") & (zone_hour.hour_start < "2020-03-01")].copy()
    means = window.groupby("zone_id").pickups.mean().to_dict()
    assert check_window(window, means, 24, 20, 0) == []

    window.loc[(window.zone_id == 237) & (window.hour_start >= "2020-02-10"), "pickups"] = 0
    issues = check_window(window, means, 24, 20, 0)
    assert [i["check"] for i in issues] == ["silent_zone"]
    assert issues[0]["zone_id"] == 237


def test_quality_does_not_flag_a_zone_quiet_because_the_whole_city_collapsed(zone_hour):
    ref = zone_hour[(zone_hour.hour_start >= "2020-01-01") & (zone_hour.hour_start < "2020-02-01")]
    means = ref.groupby("zone_id").pickups.mean().to_dict()
    window = zone_hour[(zone_hour.hour_start >= "2020-04-01") & (zone_hour.hour_start < "2020-05-01")].copy()
    smallest = min(means, key=means.get)
    window.loc[(window.zone_id == smallest) & (window.hour_start < "2020-04-03"), "pickups"] = 0
    # the city runs at ~10% of January: the smallest zone being empty for 2 days is plausible
    assert check_window(window, means, 24, 5, 0) == []

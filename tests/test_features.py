import numpy as np
import pandas as pd

from src.features import (
    create_demand_features,
    remove_incomplete_history,
)


def make_sample_data():
    """
    Build deterministic hourly demand for two stations.

    Sequential values make lag and rolling calculations easy to
    verify manually inside the tests.
    """

    hours = pd.date_range(
        "2026-04-01 00:00:00",
        periods=200,
        freq="h",
    )

    # Station A uses simple sequential values:
    # departures = 0, 1, 2, ...
    # arrivals   = 1000, 1001, 1002, ...
    station_a = pd.DataFrame(
        {
            "station_id": "A",
            "hour": hours,
            "departures": np.arange(200),
            "arrivals": np.arange(200) + 1000,
            "net_flow": 1000,
        }
    )

    # Station B uses a very different value range.
    # This helps detect accidental mixing between station histories.
    station_b = pd.DataFrame(
        {
            "station_id": "B",
            "hour": hours,
            "departures": np.arange(200) + 10000,
            "arrivals": np.arange(200) + 20000,
            "net_flow": 10000,
        }
    )

    return pd.concat(
        [station_a, station_b],
        ignore_index=True,
    )


def test_lag_features_use_past_values():
    """
    Verify that lag features point to the correct historical hours.
    """

    df = make_sample_data()
    features = create_demand_features(df)

    # April 8 00:00 is exactly 168 hours after April 1 00:00.
    # This lets us verify 1-hour, 24-hour and 168-hour lags directly.
    row = features[
        (features["station_id"] == "A")
        & (
            features["hour"]
            == pd.Timestamp("2026-04-08 00:00:00")
        )
    ].iloc[0]

    assert row["departures_lag_1"] == 167
    assert row["departures_lag_24"] == 144
    assert row["departures_lag_168"] == 0

    assert row["arrivals_lag_1"] == 1167
    assert row["arrivals_lag_24"] == 1144
    assert row["arrivals_lag_168"] == 1000


def test_rolling_features_exclude_current_hour():
    """
    Verify that rolling averages use only completed historical hours.

    The current target hour must not be included because that would
    leak information into the model.
    """

    df = make_sample_data()
    features = create_demand_features(df)

    row = features[
        (features["station_id"] == "A")
        & (
            features["hour"]
            == pd.Timestamp("2026-04-08 00:00:00")
        )
    ].iloc[0]

    # Previous 24 departure values are 144 through 167.
    assert row["departures_rolling_mean_24"] == 155.5

    # Previous 168 departure values are 0 through 167.
    assert row["departures_rolling_mean_168"] == 83.5


def test_current_target_does_not_leak_into_features():
    """
    Prove that changing the current hour's target does not change
    any predictor created for that same hour.
    """

    df = make_sample_data()

    target_time = pd.Timestamp(
        "2026-04-08 00:00:00"
    )

    original = create_demand_features(df)

    modified = df.copy()

    mask = (
        (modified["station_id"] == "A")
        & (modified["hour"] == target_time)
    )

    # Deliberately replace the current targets with extreme values.
    # Proper lag/rolling features should remain completely unchanged.
    modified.loc[mask, "departures"] = 999999
    modified.loc[mask, "arrivals"] = 999999

    changed = create_demand_features(modified)

    feature_columns = [
        "departures_lag_1",
        "departures_lag_24",
        "departures_lag_168",
        "departures_rolling_mean_24",
        "departures_rolling_mean_168",
        "arrivals_lag_1",
        "arrivals_lag_24",
        "arrivals_lag_168",
        "arrivals_rolling_mean_24",
        "arrivals_rolling_mean_168",
    ]

    original_row = original[
        (original["station_id"] == "A")
        & (original["hour"] == target_time)
    ].iloc[0]

    changed_row = changed[
        (changed["station_id"] == "A")
        & (changed["hour"] == target_time)
    ].iloc[0]

    # If any of these values changes, the current target has leaked
    # into the predictors and the feature pipeline is unsafe.
    for column in feature_columns:
        assert original_row[column] == changed_row[column]


def test_station_histories_do_not_mix():
    """
    Verify that lag calculations restart independently for each station.
    """

    df = make_sample_data()
    features = create_demand_features(df)

    first_b = features[
        features["station_id"] == "B"
    ].iloc[0]

    # Station B's first observation has no previous B observation.
    # It must therefore remain NaN instead of using Station A's last row.
    assert pd.isna(first_b["departures_lag_1"])
    assert pd.isna(first_b["arrivals_lag_1"])


def test_remove_incomplete_history():
    """
    Verify that rows without a full 168-hour history are excluded.
    """

    df = make_sample_data()
    features = create_demand_features(df)

    clean = remove_incomplete_history(features)

    # Each station contains 200 hours.
    # The first 168 cannot have complete weekly history:
    #
    # 200 - 168 = 32 usable hours per station
    # 32 × 2 stations = 64 final rows.
    assert len(clean) == 64

    # April 8 is the first timestamp with one complete week
    # of historical observations available.
    assert clean["hour"].min() == pd.Timestamp(
        "2026-04-08 00:00:00"
    )

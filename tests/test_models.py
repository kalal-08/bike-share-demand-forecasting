import pandas as pd

from src.models import (
    chronological_split,
    seasonal_naive_predictions,
)


def test_chronological_split_boundaries_and_no_overlap():
    """Protect exact train, validation, and test time boundaries."""

    hours = pd.to_datetime(
        [
            "2026-05-31 23:00:00",
            "2026-06-01 00:00:00",
            "2026-06-15 23:00:00",
            "2026-06-16 00:00:00",
            "2026-06-30 23:00:00",
            "2026-07-01 00:00:00",
        ]
    )
    df = pd.DataFrame({"hour": hours})

    train, validation, test = chronological_split(df)

    assert train["hour"].tolist() == [hours[0]]
    assert validation["hour"].tolist() == hours[1:3].tolist()
    assert test["hour"].tolist() == hours[3:5].tolist()

    split_indexes = [
        set(train.index),
        set(validation.index),
        set(test.index),
    ]
    assert split_indexes[0].isdisjoint(split_indexes[1])
    assert split_indexes[0].isdisjoint(split_indexes[2])
    assert split_indexes[1].isdisjoint(split_indexes[2])


def test_seasonal_naive_uses_one_week_lags():
    """Protect baseline mapping from each target to its 168-hour lag."""

    df = pd.DataFrame(
        {
            "departures_lag_168": [3, 8],
            "arrivals_lag_168": [5, 13],
        }
    )

    result = seasonal_naive_predictions(df)

    assert result["predicted_departures"].tolist() == [3, 8]
    assert result["predicted_arrivals"].tolist() == [5, 13]

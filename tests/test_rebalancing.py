import pandas as pd

from src.rebalancing import create_rebalancing_priorities


def test_rebalancing_directions_and_priority():
    """
    Test station-imbalance direction and priority logic.

    We create three stations for the same hour:
    - Station A should have a bike deficit.
    - Station B should have a bike surplus.
    - Station C should remain balanced.

    We also verify that the station with the largest predicted
    imbalance receives the highest attention priority.
    """

    df = pd.DataFrame(
        {
            "station_id": ["A", "B", "C"],
            "hour": pd.to_datetime(
                [
                    "2026-06-20 08:00:00",
                    "2026-06-20 08:00:00",
                    "2026-06-20 08:00:00",
                ]
            ),
            "departures": [10, 10, 10],
            "arrivals": [10, 10, 10],
        }
    )

    # Predicted station behavior:
    #
    # A: 5 arrivals - 20 departures = -15
    #    More bikes leave than arrive -> deficit pressure.
    #
    # B: 18 arrivals - 8 departures = +10
    #    More bikes arrive than leave -> surplus pressure.
    #
    # C: 10 arrivals - 10 departures = 0
    #    Expected inventory stays balanced.
    predicted_departures = [20, 8, 10]
    predicted_arrivals = [5, 18, 10]

    result = create_rebalancing_priorities(
        df,
        predicted_departures,
        predicted_arrivals,
    )

    directions = dict(
        zip(
            result["station_id"],
            result["imbalance_direction"],
        )
    )

    assert directions["A"] == "DEFICIT_PRESSURE"
    assert directions["B"] == "SURPLUS_PRESSURE"
    assert directions["C"] == "BALANCED"

    # Priority is based on the absolute predicted net flow.
    #
    # A = |-15| = 15
    # B = |+10| = 10
    # C = |0| = 0
    #
    # Therefore Station A must be ranked first.
    top_station = result.iloc[0]

    assert top_station["station_id"] == "A"
    assert top_station["priority_rank"] == 1
    assert top_station["priority_score"] == 15

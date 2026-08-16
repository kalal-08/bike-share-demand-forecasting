import numpy as np
import pandas as pd


def create_rebalancing_priorities(
    df: pd.DataFrame,
    departure_predictions,
    arrival_predictions,
) -> pd.DataFrame:
    """
    Convert demand forecasts into station-imbalance priorities.

    Negative net flow indicates deficit pressure; positive net flow
    indicates surplus pressure. Priority score ranks imbalance
    magnitude to support rebalancing attention. This signal is not
    a routing or dispatch optimizer.
    """

    result = df[
        [
            "station_id",
            "hour",
            "departures",
            "arrivals",
        ]
    ].copy()

    result["predicted_departures"] = departure_predictions
    result["predicted_arrivals"] = arrival_predictions

    # Expected change in bike inventory during the hour.
    result["predicted_net_flow"] = (
        result["predicted_arrivals"]
        - result["predicted_departures"]
    )

    # Larger absolute imbalance receives higher attention priority.
    result["priority_score"] = np.abs(
        result["predicted_net_flow"]
    )

    result["imbalance_direction"] = np.select(
        [
            result["predicted_net_flow"] < 0,
            result["predicted_net_flow"] > 0,
        ],
        [
            "DEFICIT_PRESSURE",
            "SURPLUS_PRESSURE",
        ],
        default="BALANCED",
    )

    # Rank stations independently within every hour.
    # Rank 1 represents the largest predicted imbalance.
    result["priority_rank"] = (
        result.groupby("hour")["priority_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    return result.sort_values(
        ["hour", "priority_rank"]
    ).reset_index(drop=True)

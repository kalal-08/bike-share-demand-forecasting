import pandas as pd


LAG_HOURS = [1, 24, 168]
ROLLING_WINDOWS = [24, 168]


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features from the hourly timestamp."""

    df = df.copy()

    df["hour"] = pd.to_datetime(df["hour"])

    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek

    return df


def create_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create past-only demand features for each station.

    All features are past-only. Rolling means are shifted by one
    hour, so they exclude the current target. Validation and test
    evaluation is rolling one-hour-ahead: each forecast may use
    earlier observed demand as lag input. This is not recursive
    multi-day forecasting.
    """

    df = df.copy()

    df["hour"] = pd.to_datetime(df["hour"])

    df = df.sort_values(
        ["station_id", "hour"]
    ).reset_index(drop=True)

    df = create_time_features(df)

    for target in ["departures", "arrivals"]:

        grouped = df.groupby(
            "station_id",
            sort=False,
        )[target]

        for lag in LAG_HOURS:
            df[f"{target}_lag_{lag}"] = (
                grouped.shift(lag)
            )

        # Shift before rolling so the current target cannot enter
        # its own rolling-mean predictors.
        shifted = df.groupby(
            "station_id",
            sort=False,
        )[target].shift(1)

        for window in ROLLING_WINDOWS:
            df[f"{target}_rolling_mean_{window}"] = (
                shifted
                .groupby(df["station_id"])
                .rolling(
                    window=window,
                    min_periods=window,
                )
                .mean()
                .reset_index(
                    level=0,
                    drop=True,
                )
            )

    return df


def remove_incomplete_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove rows without the full 168-hour feature history.
    """

    required_features = [
        "departures_lag_168",
        "arrivals_lag_168",
        "departures_rolling_mean_168",
        "arrivals_rolling_mean_168",
    ]

    return (
        df.dropna(subset=required_features)
        .reset_index(drop=True)
    )

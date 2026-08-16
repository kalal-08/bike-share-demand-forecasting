import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Fixed chronological boundaries.
# Forecasting data must never be randomly shuffled because that
# could allow future observations to influence model training.
TRAIN_END = pd.Timestamp("2026-06-01 00:00:00")
VALIDATION_END = pd.Timestamp("2026-06-16 00:00:00")
TEST_END = pd.Timestamp("2026-07-01 00:00:00")


def chronological_split(df: pd.DataFrame):
    """
    Split station-hour observations in chronological order.

    Usable feature data begins on April 8 because the first
    168 hours are required to construct one-week lag features.
    """

    df = df.copy()
    df["hour"] = pd.to_datetime(df["hour"])

    train = df[
        df["hour"] < TRAIN_END
    ].copy()

    validation = df[
        (df["hour"] >= TRAIN_END)
        & (df["hour"] < VALIDATION_END)
    ].copy()

    test = df[
        (df["hour"] >= VALIDATION_END)
        & (df["hour"] < TEST_END)
    ].copy()

    return train, validation, test


def seasonal_naive_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple one-week seasonal baseline.

    The prediction for an hour is the observed demand at the
    same station and same hour exactly 168 hours earlier.
    """

    result = df.copy()

    result["predicted_departures"] = (
        result["departures_lag_168"]
    )

    result["predicted_arrivals"] = (
        result["arrivals_lag_168"]
    )

    return result


def get_model_features(target: str):
    """
    Return the predictors used for one demand target.

    Departure and arrival models are trained separately so each
    model uses the historical behavior of the quantity it predicts.
    """

    if target not in {"departures", "arrivals"}:
        raise ValueError(
            "target must be either 'departures' or 'arrivals'"
        )

    categorical_features = [
        "station_id",
        "hour_of_day",
        "day_of_week",
    ]

    numeric_features = [
        f"{target}_lag_1",
        f"{target}_lag_24",
        f"{target}_lag_168",
        f"{target}_rolling_mean_24",
        f"{target}_rolling_mean_168",
    ]

    return categorical_features, numeric_features


def build_poisson_model(target: str):
    """
    Build a Poisson-regression forecasting pipeline.

    Categorical station/time variables are one-hot encoded,
    while lag and rolling-demand features are standardized.
    """

    categorical_features, numeric_features = (
        get_model_features(target)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                PoissonRegressor(
                    alpha=1.0,
                    max_iter=500,
                ),
            ),
        ]
    )

    return model


def train_poisson_model(
    train: pd.DataFrame,
    target: str,
):
    """Fit one Poisson model on the training period only."""

    categorical_features, numeric_features = (
        get_model_features(target)
    )

    feature_columns = (
        categorical_features + numeric_features
    )

    model = build_poisson_model(target)

    model.fit(
        train[feature_columns],
        train[target],
    )

    return model


def predict_poisson(
    model,
    df: pd.DataFrame,
    target: str,
):
    """Generate Poisson forecasts for a given time period."""

    categorical_features, numeric_features = (
        get_model_features(target)
    )

    feature_columns = (
        categorical_features + numeric_features
    )

    return model.predict(
        df[feature_columns]
    )


def build_random_forest_model(target: str):
    """
    Build a nonlinear Random Forest demand model.

    Categorical station/time variables are one-hot encoded.
    Random Forest does not require numeric feature scaling.
    """

    categorical_features, numeric_features = (
        get_model_features(target)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=18,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def train_random_forest_model(
    train: pd.DataFrame,
    target: str,
):
    """Train one Random Forest model using training data only."""

    categorical_features, numeric_features = (
        get_model_features(target)
    )

    feature_columns = (
        categorical_features + numeric_features
    )

    model = build_random_forest_model(target)

    model.fit(
        train[feature_columns],
        train[target],
    )

    return model


def predict_random_forest(
    model,
    df: pd.DataFrame,
    target: str,
):
    """Generate Random Forest demand forecasts."""

    categorical_features, numeric_features = (
        get_model_features(target)
    )

    feature_columns = (
        categorical_features + numeric_features
    )

    predictions = model.predict(
        df[feature_columns]
    )

    # Demand cannot be negative.
    return predictions.clip(min=0)


def combine_train_validation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine training and validation periods after model selection.

    Validation data is only added after Random Forest has been
    selected, so the untouched test period remains the final
    independent evaluation set.
    """

    combined = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    return combined.sort_values(
        ["station_id", "hour"]
    ).reset_index(drop=True)

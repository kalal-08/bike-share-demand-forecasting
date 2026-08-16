import numpy as np


def mae(y_true, y_pred):
    """Mean Absolute Error: average absolute forecast error."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(
        np.mean(np.abs(y_true - y_pred))
    )


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.

    Larger mistakes receive more penalty than they do under MAE.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error.

    Measures total absolute forecast error relative to the
    total observed demand.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return float(
        np.sum(
            np.abs(y_true - y_pred)
        )
        / denominator
        * 100
    )


def evaluate_forecast(y_true, y_pred):
    """Return the project's three forecasting metrics."""

    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
    }

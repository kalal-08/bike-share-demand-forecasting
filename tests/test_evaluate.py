import numpy as np
import pytest

from src.evaluate import evaluate_forecast, wape


def test_forecast_metrics_and_zero_demand_wape():
    """Protect metric formulas and undefined zero-demand WAPE behavior."""

    metrics = evaluate_forecast(
        y_true=[1, 2, 3],
        y_pred=[1, 4, 2],
    )

    assert metrics["MAE"] == pytest.approx(1.0)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["WAPE"] == pytest.approx(50.0)
    assert np.isnan(wape([0, 0], [1, 2]))

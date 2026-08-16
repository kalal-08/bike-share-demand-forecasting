from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sqlalchemy import text

from src.data import get_database_engine


METRICS_PATH = Path("outputs/model_metrics.csv")
PREDICTIONS_PATH = Path(
    "data/processed/rebalancing_predictions.csv"
)
FIGURE_DIR = Path("outputs/figures")


def load_plot_data():
    """
    Load model metrics and final test-period predictions used
    by the project figures.
    """

    metrics = pd.read_csv(METRICS_PATH)

    # Station IDs are identifiers, so preserve them as strings
    # even though Citi Bike IDs often look numeric.
    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        dtype={"station_id": "string"},
        parse_dates=["hour"],
    )

    return metrics, predictions


def get_top_training_station():
    """
    Return the busiest station selected from training data only.

    Using training activity avoids choosing a station afterward
    simply because it produces a visually convenient test result.
    """

    engine = get_database_engine()

    query = text(
        """
        SELECT station_id
        FROM top_stations_train
        ORDER BY total_activity DESC
        LIMIT 1
        """
    )

    with engine.connect() as connection:
        station_id = connection.execute(
            query
        ).scalar_one()

    engine.dispose()

    return str(station_id)


def plot_model_comparison(metrics):
    """
    Compare validation WAPE across the three forecasting models.
    Lower WAPE indicates better aggregate forecasting accuracy.
    """

    validation = metrics[
        metrics["stage"] == "validation"
    ].copy()

    model_order = [
        "Seasonal Naive",
        "Poisson",
        "Random Forest",
    ]

    departures = (
        validation[
            validation["target"] == "departures"
        ]
        .set_index("model")
        .loc[model_order]["WAPE"]
    )

    arrivals = (
        validation[
            validation["target"] == "arrivals"
        ]
        .set_index("model")
        .loc[model_order]["WAPE"]
    )

    x = np.arange(len(model_order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        departures,
        width,
        label="Departures",
    )

    ax.bar(
        x + width / 2,
        arrivals,
        width,
        label="Arrivals",
    )

    ax.set_title(
        "Validation WAPE by Model (Lower Is Better)"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("WAPE (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_order)
    ax.legend()

    # Display the measured error directly above each bar so the
    # figure remains understandable without reading raw logs.
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f%%",
            padding=3,
        )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR / "model_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_actual_vs_predicted(
    predictions,
    station_id,
):
    """
    Show actual and predicted departures for a fixed 72-hour
    window at the busiest training-period station.
    """

    station_data = (
        predictions[
            predictions["station_id"] == station_id
        ]
        .sort_values("hour")
        .head(72)
        .copy()
    )

    if station_data.empty:
        raise ValueError(
            f"No prediction rows found for station {station_id}"
        )

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        station_data["hour"],
        station_data["departures"],
        label="Actual departures",
    )

    ax.plot(
        station_data["hour"],
        station_data["predicted_departures"],
        label="Predicted departures",
    )

    ax.set_title(
        f"Actual vs Predicted Hourly Departures "
        f"— Station {station_id}"
    )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Bike departures")
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = (
        FIGURE_DIR / "actual_vs_predicted.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_rebalancing_priorities(predictions):
    """
    Visualize the highest-priority station imbalances for the
    final hour of the test period.

    Negative net flow indicates deficit pressure, while positive
    net flow indicates surplus pressure.
    """

    latest_hour = predictions["hour"].max()

    latest = (
        predictions[
            predictions["hour"] == latest_hour
        ]
        .sort_values(
            "priority_rank"
        )
        .head(10)
        .sort_values(
            "predicted_net_flow"
        )
        .copy()
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    deficit_color = "#D55E00"
    surplus_color = "#0072B2"
    colors = np.where(
        latest["predicted_net_flow"] < 0,
        deficit_color,
        surplus_color,
    )

    ax.barh(
        latest["station_id"],
        latest["predicted_net_flow"],
        color=colors,
    )

    # Zero separates expected deficit pressure from surplus pressure.
    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "Predicted Station Imbalance — Final Test Hour"
    )

    ax.set_xlabel(
        "Predicted net flow "
        "(arrivals - departures)"
    )

    ax.set_ylabel("Station ID")

    ax.legend(
        handles=[
            Patch(
                color=deficit_color,
                label="Deficit Pressure",
            ),
            Patch(
                color=surplus_color,
                label="Surplus Pressure",
            ),
        ]
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR / "rebalancing_priorities.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_error_analysis(predictions):
    """Plot final-test absolute forecast error by hour of day."""

    errors = predictions.copy()
    errors["departure_abs_error"] = np.abs(
        errors["departures"]
        - errors["predicted_departures"]
    )
    errors["arrival_abs_error"] = np.abs(
        errors["arrivals"]
        - errors["predicted_arrivals"]
    )
    errors["hour_of_day"] = errors["hour"].dt.hour

    hourly_errors = errors.groupby("hour_of_day")[
        ["departure_abs_error", "arrival_abs_error"]
    ].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        hourly_errors.index,
        hourly_errors["departure_abs_error"],
        marker="o",
        label="Departures",
    )
    ax.plot(
        hourly_errors.index,
        hourly_errors["arrival_abs_error"],
        marker="o",
        label="Arrivals",
    )

    ax.set_title("Final Test MAE by Hour of Day")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(
        "Mean absolute error (bikes per station-hour)"
    )
    ax.set_xticks(range(24))
    ax.legend()

    fig.tight_layout()

    output_path = FIGURE_DIR / "error_analysis.png"
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    highest_departure_hour = int(
        hourly_errors["departure_abs_error"].idxmax()
    )
    highest_arrival_hour = int(
        hourly_errors["arrival_abs_error"].idxmax()
    )

    print(f"Saved: {output_path}")
    print(
        "Highest departure MAE hour: "
        f"{highest_departure_hour}"
    )
    print(
        "Highest arrival MAE hour: "
        f"{highest_arrival_hour}"
    )


def main():
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics, predictions = load_plot_data()

    top_station = get_top_training_station()

    print(
        f"Training-period top station: {top_station}"
    )

    plot_model_comparison(metrics)

    plot_actual_vs_predicted(
        predictions,
        top_station,
    )

    plot_rebalancing_priorities(
        predictions
    )

    plot_error_analysis(predictions)

    print("\nFigure generation complete.")


if __name__ == "__main__":
    main()

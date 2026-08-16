from pathlib import Path

import pandas as pd

from src.data import load_hourly_demand
from src.evaluate import evaluate_forecast
from src.features import (
    create_demand_features,
    remove_incomplete_history,
)
from src.models import (
    chronological_split,
    predict_poisson,
    predict_random_forest,
    seasonal_naive_predictions,
    train_poisson_model,
    train_random_forest_model,
    combine_train_validation,
)
from src.rebalancing import create_rebalancing_priorities


def print_metrics(title, metrics):
    """Print forecast metrics in a consistent readable format."""

    print(f"\n{title}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")


def main():
    hourly = load_hourly_demand()

    features = create_demand_features(hourly)

    # A full week (168 hours) of history is needed for our
    # longest lag and rolling feature, so incomplete rows
    # from the beginning of each station's history are removed.
    features = remove_incomplete_history(features)

    # Time-series data must remain ordered. Randomly shuffling
    # rows could allow future demand patterns into training.
    train, validation, test = chronological_split(features)

    print("Chronological split")
    print("-------------------")

    print(f"Train rows:      {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows:       {len(test):,}")

    print(
        f"Train period: "
        f"{train['hour'].min()} -> {train['hour'].max()}"
    )

    print(
        f"Validation period: "
        f"{validation['hour'].min()} -> "
        f"{validation['hour'].max()}"
    )

    print(
        f"Test period: "
        f"{test['hour'].min()} -> "
        f"{test['hour'].max()}"
    )

    # A 168-hour lag is the same station-hour one week earlier,
    # providing a meaningful seasonal baseline for ML models.
    baseline = seasonal_naive_predictions(validation)

    baseline_departure_metrics = evaluate_forecast(
        baseline["departures"],
        baseline["predicted_departures"],
    )

    baseline_arrival_metrics = evaluate_forecast(
        baseline["arrivals"],
        baseline["predicted_arrivals"],
    )

    print("\nSeasonal-naive validation results")
    print("---------------------------------")

    print_metrics(
        "Departures",
        baseline_departure_metrics,
    )

    print_metrics(
        "Arrivals",
        baseline_arrival_metrics,
    )

    # Bike departures and arrivals are count-valued targets,
    # making Poisson Regression a useful statistical benchmark.
    print("\nPoisson-regression validation results")
    print("-------------------------------------")

    departure_poisson_model = train_poisson_model(
        train,
        target="departures",
    )

    arrival_poisson_model = train_poisson_model(
        train,
        target="arrivals",
    )

    poisson_departure_predictions = predict_poisson(
        departure_poisson_model,
        validation,
        target="departures",
    )

    poisson_arrival_predictions = predict_poisson(
        arrival_poisson_model,
        validation,
        target="arrivals",
    )

    poisson_departure_metrics = evaluate_forecast(
        validation["departures"],
        poisson_departure_predictions,
    )

    poisson_arrival_metrics = evaluate_forecast(
        validation["arrivals"],
        poisson_arrival_predictions,
    )

    print_metrics(
        "Departures",
        poisson_departure_metrics,
    )

    print_metrics(
        "Arrivals",
        poisson_arrival_metrics,
    )

    print("\nRandom-Forest validation results")
    print("--------------------------------")

    departure_rf_model = train_random_forest_model(
        train,
        target="departures",
    )

    arrival_rf_model = train_random_forest_model(
        train,
        target="arrivals",
    )

    rf_departure_predictions = predict_random_forest(
        departure_rf_model,
        validation,
        target="departures",
    )

    rf_arrival_predictions = predict_random_forest(
        arrival_rf_model,
        validation,
        target="arrivals",
    )

    rf_departure_metrics = evaluate_forecast(
        validation["departures"],
        rf_departure_predictions,
    )

    rf_arrival_metrics = evaluate_forecast(
        validation["arrivals"],
        rf_arrival_predictions,
    )

    print_metrics(
        "Departures",
        rf_departure_metrics,
    )

    print_metrics(
        "Arrivals",
        rf_arrival_metrics,
    )

    # Model selection is based ONLY on validation performance.
    # The test period remains untouched until a final model has
    # been selected.
    print("\nValidation comparison")
    print("---------------------")

    print("\nDEPARTURES")
    print(
        f"{'Model':<20}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'WAPE':>12}"
    )

    print(
        f"{'Seasonal Naive':<20}"
        f"{baseline_departure_metrics['MAE']:>10.3f}"
        f"{baseline_departure_metrics['RMSE']:>10.3f}"
        f"{baseline_departure_metrics['WAPE']:>11.3f}%"
    )

    print(
        f"{'Poisson':<20}"
        f"{poisson_departure_metrics['MAE']:>10.3f}"
        f"{poisson_departure_metrics['RMSE']:>10.3f}"
        f"{poisson_departure_metrics['WAPE']:>11.3f}%"
    )

    print(
        f"{'Random Forest':<20}"
        f"{rf_departure_metrics['MAE']:>10.3f}"
        f"{rf_departure_metrics['RMSE']:>10.3f}"
        f"{rf_departure_metrics['WAPE']:>11.3f}%"
    )

    print("\nARRIVALS")
    print(
        f"{'Model':<20}"
        f"{'MAE':>10}"
        f"{'RMSE':>10}"
        f"{'WAPE':>12}"
    )

    print(
        f"{'Seasonal Naive':<20}"
        f"{baseline_arrival_metrics['MAE']:>10.3f}"
        f"{baseline_arrival_metrics['RMSE']:>10.3f}"
        f"{baseline_arrival_metrics['WAPE']:>11.3f}%"
    )

    print(
        f"{'Poisson':<20}"
        f"{poisson_arrival_metrics['MAE']:>10.3f}"
        f"{poisson_arrival_metrics['RMSE']:>10.3f}"
        f"{poisson_arrival_metrics['WAPE']:>11.3f}%"
    )

    print(
        f"{'Random Forest':<20}"
        f"{rf_arrival_metrics['MAE']:>10.3f}"
        f"{rf_arrival_metrics['RMSE']:>10.3f}"
        f"{rf_arrival_metrics['WAPE']:>11.3f}%"
    )

    # Random Forest won the validation comparison, so it is now
    # retrained on train + validation before one untouched test.

    print("\nFinal Random-Forest test evaluation")
    print("-----------------------------------")

    final_train = combine_train_validation(
        train,
        validation,
    )

    print(
        f"Final training rows: {len(final_train):,}"
    )

    print(
        f"Final training period: "
        f"{final_train['hour'].min()} -> "
        f"{final_train['hour'].max()}"
    )

    final_departure_model = train_random_forest_model(
        final_train,
        target="departures",
    )

    final_arrival_model = train_random_forest_model(
        final_train,
        target="arrivals",
    )

    final_departure_predictions = predict_random_forest(
        final_departure_model,
        test,
        target="departures",
    )

    final_arrival_predictions = predict_random_forest(
        final_arrival_model,
        test,
        target="arrivals",
    )

    # Rolling one-hour-ahead forecasts may use earlier observed demand;
    # test data is never used for model selection.
    final_departure_metrics = evaluate_forecast(
        test["departures"],
        final_departure_predictions,
    )

    final_arrival_metrics = evaluate_forecast(
        test["arrivals"],
        final_arrival_predictions,
    )

    print_metrics(
        "Test Departures",
        final_departure_metrics,
    )

    print_metrics(
        "Test Arrivals",
        final_arrival_metrics,
    )

    # Store the verified validation and final-test results so
    # figures can be regenerated without manually copying values.

    metrics_rows = [
        {
            "stage": "validation",
            "model": "Seasonal Naive",
            "target": "departures",
            **baseline_departure_metrics,
        },
        {
            "stage": "validation",
            "model": "Seasonal Naive",
            "target": "arrivals",
            **baseline_arrival_metrics,
        },
        {
            "stage": "validation",
            "model": "Poisson",
            "target": "departures",
            **poisson_departure_metrics,
        },
        {
            "stage": "validation",
            "model": "Poisson",
            "target": "arrivals",
            **poisson_arrival_metrics,
        },
        {
            "stage": "validation",
            "model": "Random Forest",
            "target": "departures",
            **rf_departure_metrics,
        },
        {
            "stage": "validation",
            "model": "Random Forest",
            "target": "arrivals",
            **rf_arrival_metrics,
        },
        {
            "stage": "test",
            "model": "Random Forest",
            "target": "departures",
            **final_departure_metrics,
        },
        {
            "stage": "test",
            "model": "Random Forest",
            "target": "arrivals",
            **final_arrival_metrics,
        },
    ]

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_path = Path(
        "outputs/model_metrics.csv"
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        f"\nModel metrics saved to: {metrics_path}"
    )

    # predicted_net_flow = arrivals - departures
    # Negative means deficit pressure; positive means surplus pressure.
    # Rankings guide attention, not routing or dispatch decisions.

    print("\nGenerating station-imbalance priorities...")
    print("-----------------------------------")

    rebalancing = create_rebalancing_priorities(
        test,
        final_departure_predictions,
        final_arrival_predictions,
    )

    # Keep the processed output separate from the raw Citi Bike
    # files so downstream analysis can use the predictions directly.
    output_dir = Path("data/processed")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "rebalancing_predictions.csv"
    )

    rebalancing.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Rebalancing rows: {len(rebalancing):,}"
    )

    print(
        f"Period: "
        f"{rebalancing['hour'].min()} -> "
        f"{rebalancing['hour'].max()}"
    )

    print(
        f"Saved to: {output_path}"
    )

    latest_hour = rebalancing["hour"].max()

    latest_priorities = (
        rebalancing[
            rebalancing["hour"] == latest_hour
        ]
        .sort_values(
            "priority_rank"
        )
        .head(5)
    )

    print(
        f"\nTop rebalancing priorities "
        f"for {latest_hour}"
    )
    print(
        "-----------------------------------"
    )

    print(
        latest_priorities[
            [
                "station_id",
                "predicted_departures",
                "predicted_arrivals",
                "predicted_net_flow",
                "priority_score",
                "priority_rank",
                "imbalance_direction",
            ]
        ].to_string(index=False)
    )

if __name__ == "__main__":
    main()

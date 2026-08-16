# Reference Audit

Related work reviewed:
https://github.com/DanieleGammelli/variational-poisson-rnn

## Problem
Forecast bike-sharing demand and use demand forecasts to support inventory decisions.

## Reference flow
Historical demand
→ temporal aggregation
→ forecasting model
→ demand predictions
→ inventory decision

## Ideas retained
- Station-level demand forecasting
- Time-based aggregation
- Forecast evaluation
- Forecast-to-decision connection

## Components not copied
- VP-RNN implementation
- MOVP-RNN implementation
- Original training scripts
- Original inventory decision implementation
- Saved models/results
- Original README/code structure

## Implemented direction
Official Citi Bike raw ZIPs
→ chunked Python validation
→ PostgreSQL COPY ingestion
→ SQL station-hour aggregation
→ training-only top-station selection
→ past-only feature engineering
→ chronological validation
→ Seasonal Naive / Poisson / Random Forest comparison
→ final unseen test evaluation
→ station imbalance prioritization
→ error analysis

This project does not implement or reproduce VP-RNN or MOVP-RNN.
It uses an independent data pipeline, feature engineering process,
classical forecasting models, chronological evaluation design, and
station-imbalance prioritization signal.

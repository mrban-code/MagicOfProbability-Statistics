# PM2.5 Prediction Model Supplement

This folder contains reproducible prediction experiments for the PM2.5 paper.

## Main command

```bash
'/Users/chenlichong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3' run_prediction_models.py
```

## Outputs

- `data/prediction_feature_dataset.csv`: daily lag feature dataset.
- `data/daily_forecast_test_predictions.csv`: test predictions for same-day PM2.5 from previous-day information.
- `data/trend_anchor_feature_dataset.csv`: 7-day historical-window feature dataset.
- `data/trend_forecast_test_predictions.csv`: 1-7 day trend predictions.
- `results/*.csv`: metrics, validation logs, coefficients, and classification tables.
- `results/stat_*.csv`: bootstrap confidence intervals, paired error tests, residual bias tests, conformal intervals, and Wilson accuracy intervals.
- `figures/*.png`: charts used in the Word report.
- `PM25_prediction_model_report.docx`: paper-ready prediction-model section.

The split is chronological: training/validation before 2025-05-22, final test from 2025-05-22 to 2026-05-21.
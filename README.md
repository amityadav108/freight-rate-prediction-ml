# Freight Rate Prediction ML

A production-oriented machine learning solution for predicting freight load rates from historical shipment, route, equipment, market, and quote data.

This project was developed as part of a Machine Learning Engineer assessment and focuses on building a reproducible prediction pipeline that generalizes from historical freight loads to future unseen loads.

## Overview

The objective is to:

1. Train and validate a regression model using historical labeled freight loads.
2. Build a validation strategy that reflects the future-prediction setting.
3. Engineer meaningful route, geographic, temporal, and pricing features.
4. Train a final model using the available development data.
5. Generate predictions for 12,000 unseen validation loads.
6. Generate the required December 2025 prediction series.
7. Validate the final outputs using the provided scoring script.

## Key Approach

### Temporal validation

Because the labeled data covers historical months and the validation data represents future dates, a random train/test split was avoided.

Instead, the model was evaluated using expanding-window temporal validation:

* August 2025 holdout
* September 2025 holdout
* October 2025 holdout

This better represents the real-world deployment scenario where historical observations are available but future observations are not.

### Feature engineering

The pipeline creates:

* Route-level categorical features
* Geographic distance features
* Supplied-distance versus geographic-distance ratios
* Calendar and seasonal features
* Cyclical date encodings
* Log-transformed distance and weight
* Weight-per-mile features
* Quote-signal-per-mile features
* Quote and market interaction features
* Missing-value indicator features

The geographic features are particularly useful for handling validation routes containing cities that were not directly observed as categorical values during training.

### Model

The final model uses `CatBoostRegressor`.

CatBoost was selected because the dataset combines numerical and categorical variables and contains nonlinear relationships between route, distance, equipment, weight, market conditions, and pricing signals.

Final configuration:

* Depth: 9
* Learning rate: 0.05
* L2 regularization: 8
* Iterations: 130
* Random seed: 42

## Validation Results

| Holdout        |        MAE |       RMSE |        R² |      MAPE |
| -------------- | ---------: | ---------: | --------: | --------: |
| August 2025    |     144.21 |     619.78 |     0.823 |     7.35% |
| September 2025 |     118.54 |     618.31 |     0.835 |     5.51% |
| October 2025   |     123.11 |     648.26 |     0.820 |     7.48% |
| **Mean**       | **128.62** | **628.78** | **0.826** | **6.78%** |

MAE is used as the primary internal metric because it provides a direct interpretation of average prediction error in dollars while being less sensitive to extreme target values than RMSE.

## Data Quality

The analysis identified several data-quality considerations:

* Missing `weight` values
* Missing `market_index` values
* Previously unseen pickup and delivery cities in validation
* Extreme freight-rate observations

Missing numerical values are handled using training-data statistics together with missingness indicators.

Future validation data is never used to calculate training preprocessing statistics.

For unseen geographic locations, coordinate-based features provide a representation that is not dependent on the exact categorical city value appearing during training.

## Reproducibility

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the complete training and prediction workflow:

```bash
python src/train.py
```

Validate the generated outputs:

```bash
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv
```

The scorer validates:

* Exactly 12,000 validation predictions
* Required `load_id` values
* Prediction column format
* Positive predicted rates
* All 31 December dates
* Required fixed December scenario
* Valid December predictions

It also generates:

```text
scorer_results/candidate_december.png
```

## Repository Structure

```text
freight-rate-prediction-ml/
│
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
│
├── src/
│   ├── __init__.py
│   ├── features.py
│   └── train.py
│
├── reports/
│   ├── temporal_cv_metrics.csv
│   └── Freight_Rate_ML_Assessment_Report.pdf
│
├── scorer_results/
│   └── candidate_december.png
│
├── validation_predictions.csv
├── score.py
├── requirements.txt
├── LOOM_SCRIPT.md
├── README.md
└── .gitignore
```

## Deliverables

The repository contains:

* Complete training and inference code
* Feature engineering pipeline
* Temporal cross-validation results
* Final 12,000-row prediction file
* December prediction inputs and outputs
* Required December prediction chart
* Assessment report
* Reproducible dependency specification
* Loom walkthrough script

## Engineering Principles

The implementation emphasizes:

* Reproducibility
* Leakage-aware validation
* Temporal generalization
* Robust handling of missing data
* Generalization to unseen categorical values
* Clear separation of feature engineering and training
* Deterministic model configuration
* Explicit output validation

## Final Prediction

The final submission file is:

```text
validation_predictions.csv
```

with the required schema:

```text
load_id,predicted_rate
```

containing predictions for all 12,000 validation loads.

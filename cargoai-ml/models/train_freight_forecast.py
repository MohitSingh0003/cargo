"""
train_freight_forecast.py

Trains a gradient-boosted regression model that predicts freight rate
(USD/ton) from route, cargo type, fuel price index and time-derived
features (day-of-year seasonality, days-since-start trend, lag features).

This replaces the moving-average / linear-trend placeholder described in
the README with a real, evaluated, persisted model while keeping the
same public function signature in models/freight_forecast.py, so
cargoai-backend/src/services/mlClient.js does not need to change.

Run:
    python training/train_freight_forecast.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "historical_rates.csv"
MODEL_PATH = ROOT / "models" / "trained" / "freight_forecast_model.joblib"
METRICS_PATH = ROOT / "models" / "trained" / "freight_forecast_metrics.json"

CATEGORICAL = ["route", "cargo_type"]
NUMERIC = ["fuel_price_index", "day_of_year_sin", "day_of_year_cos", "days_since_start",
           "rate_lag_7", "rate_lag_30"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "cargo_type", "date"])

    doy = df["date"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

    start = df["date"].min()
    df["days_since_start"] = (df["date"] - start).dt.days

    # lag features per route/cargo group (helps the model pick up momentum,
    # which a plain moving average captures crudely but a GBM can weight
    # against fuel price and seasonality at the same time)
    grp = df.groupby(["route", "cargo_type"])["freight_rate_usd_per_ton"]
    df["rate_lag_7"] = grp.shift(7)
    df["rate_lag_30"] = grp.shift(30)
    df = df.dropna(subset=["rate_lag_7", "rate_lag_30"])
    return df


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_time_features(df)
    df = df.reset_index(drop=True)

    X = df[CATEGORICAL + NUMERIC]
    y = df["freight_rate_usd_per_ton"]

    # time-respecting split: last 15% of each route/cargo series held out
    cutoff = df["days_since_start"].quantile(0.85)
    train_idx = df["days_since_start"] <= cutoff
    X_train, X_test = X[train_idx], X[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx]

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = {
        "mae_usd_per_ton": round(mean_absolute_error(y_test, preds), 3),
        "mape": round(mean_absolute_percentage_error(y_test, preds), 4),
        "r2": round(r2_score(y_test, preds), 4),
        "n_train": int(train_idx.sum()),
        "n_test": int((~train_idx).sum()),
    }
    print("Freight forecast model evaluation:", json.dumps(metrics, indent=2))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"saved model -> {MODEL_PATH}")

    # also persist the last known lag/fuel context per (route, cargo_type)
    # so the serving code can build features for future dates without
    # needing the full training set in memory
    latest = (
        df.sort_values("date")
        .groupby(["route", "cargo_type"])
        .tail(1)[["route", "cargo_type", "date", "freight_rate_usd_per_ton",
                   "fuel_price_index", "rate_lag_7", "days_since_start"]]
    )
    latest_path = ROOT / "models" / "trained" / "freight_forecast_context.json"
    latest_records = latest.assign(date=latest["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
    latest_path.write_text(json.dumps(latest_records, indent=2))
    print(f"saved serving context -> {latest_path}")


if __name__ == "__main__":
    main()

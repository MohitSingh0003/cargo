"""
models/freight_forecast.py

Serves freight-rate forecasts from the trained GradientBoostingRegressor
in models/trained/freight_forecast_model.joblib (see
training/train_freight_forecast.py).

Public entry point keeps the same shape as the original placeholder so
cargoai-backend/src/services/mlClient.js and app.py do not need to change:

    forecast(route: str, cargo_type: str, horizon_days: int = 30) -> dict

If the trained artifact is missing (e.g. a fresh clone before anyone has
run the training scripts), this falls back to the original moving-average
+ linear-trend estimate so the service still boots and responds.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "trained" / "freight_forecast_model.joblib"
CONTEXT_PATH = ROOT / "models" / "trained" / "freight_forecast_context.json"
HISTORICAL_CSV = ROOT / "data" / "historical_rates.csv"


@lru_cache(maxsize=1)
def _load_model():
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None


@lru_cache(maxsize=1)
def _load_context():
    try:
        return {
            (row["route"], row["cargo_type"]): row
            for row in json.loads(CONTEXT_PATH.read_text())
        }
    except FileNotFoundError:
        return {}


def _fallback_moving_average(route: str, cargo_type: str, horizon_days: int) -> dict:
    """Original placeholder behaviour, kept as a safety net."""
    df = pd.read_csv(HISTORICAL_CSV)
    subset = df[(df["route"] == route) & (df["cargo_type"] == cargo_type)].sort_values("date")
    if subset.empty:
        subset = df[df["cargo_type"] == cargo_type].sort_values("date")
    rates = subset["freight_rate_usd_per_ton"].tail(30).to_numpy()
    if len(rates) == 0:
        rates = df["freight_rate_usd_per_ton"].to_numpy()

    moving_avg = float(np.mean(rates[-14:])) if len(rates) >= 14 else float(np.mean(rates))
    trend = float(np.polyfit(range(len(rates)), rates, 1)[0]) if len(rates) > 1 else 0.0

    last_date = pd.to_datetime(subset["date"]).max() if not subset.empty else pd.Timestamp.today()
    predictions = []
    for step in range(1, horizon_days + 1):
        predictions.append({
            "date": (last_date + timedelta(days=step)).date().isoformat(),
            "predicted_rate_usd_per_ton": round(moving_avg + trend * step, 2),
        })
    return {
        "route": route,
        "cargo_type": cargo_type,
        "model": "moving_average_fallback",
        "predictions": predictions,
    }


def is_trained_model_loaded() -> bool:
    """True if the trained GBM artifact loaded successfully (vs. fallback)."""
    return _load_model() is not None


def forecast(route: str, cargo_type: str, horizon_days: int = 30) -> dict:
    """
    Forecast freight rate (USD/ton) for `route` + `cargo_type` over the
    next `horizon_days` days.

    Returns:
        {
          "route": str,
          "cargo_type": str,
          "model": "gradient_boosting_regressor",
          "predictions": [{"date": "YYYY-MM-DD", "predicted_rate_usd_per_ton": float}, ...]
        }
    """
    model = _load_model()
    if model is None:
        return _fallback_moving_average(route, cargo_type, horizon_days)

    context = _load_context()
    ctx = context.get((route, cargo_type))
    if ctx is None:
        # unseen route/cargo combo -> fall back to averages across cargo type
        return _fallback_moving_average(route, cargo_type, horizon_days)

    last_date = pd.to_datetime(ctx["date"])
    fuel_price = ctx["fuel_price_index"]
    rate_lag_7 = ctx["rate_lag_7"]
    rate_lag_30 = ctx["freight_rate_usd_per_ton"]  # most recent actual, used as the 30-day lag proxy

    rows = []
    running_rate = ctx["freight_rate_usd_per_ton"]
    for step in range(1, horizon_days + 1):
        d = last_date + timedelta(days=step)
        doy = d.dayofyear
        rows.append({
            "route": route,
            "cargo_type": cargo_type,
            "fuel_price_index": fuel_price,
            "day_of_year_sin": np.sin(2 * np.pi * doy / 365.25),
            "day_of_year_cos": np.cos(2 * np.pi * doy / 365.25),
            "days_since_start": ctx.get("days_since_start", 0) + step,
            "rate_lag_7": rate_lag_7 if step <= 7 else running_rate,
            "rate_lag_30": rate_lag_30,
        })

    features = pd.DataFrame(rows)
    preds = model.predict(features)

    predictions = []
    for step, p in enumerate(preds, start=1):
        d = (last_date + timedelta(days=step)).date().isoformat()
        predictions.append({"date": d, "predicted_rate_usd_per_ton": round(float(p), 2)})
    running_rate = float(preds[-1])

    return {
        "route": route,
        "cargo_type": cargo_type,
        "model": "gradient_boosting_regressor",
        "predictions": predictions,
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(forecast("Singapore-Rotterdam", "container", horizon_days=7))

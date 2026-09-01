import os
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "historical_rates.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "freight_model.joblib")

LAGS = [1, 2, 3, 4]


def create_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["route", "date"])

    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby("route")["rate_usd_per_mt"].shift(lag)

    df["rolling_mean_4"] = (
        df.groupby("route")["rate_usd_per_mt"]
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )

    df["rolling_std_4"] = (
        df.groupby("route")["rate_usd_per_mt"]
        .transform(lambda x: x.shift(1).rolling(4).std())
    )

    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    return df


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Freight model not found. Run training/train_forecast.py first."
        )

    return joblib.load(MODEL_PATH)


def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def predict_forecast(route: str, periods: int = 4):
    df = load_data()

    route_df = (
        df[df["route"] == route]
        .sort_values("date")
        .copy()
    )

    if route_df.empty:
        return {
            "route": route,
            "forecast": [],
            "trend": "UNKNOWN",
            "message": "No historical data available for this route."
        }

    model = load_model()

    rates = route_df["rate_usd_per_mt"].tolist()
    dates = route_df["date"].tolist()

    predictions = []

    last_date = dates[-1]

    for i in range(periods):
        if len(rates) < 4:
            break

        recent = rates[-4:]

        prediction_features = pd.DataFrame([{
            "lag_1": recent[-1],
            "lag_2": recent[-2],
            "lag_3": recent[-3],
            "lag_4": recent[-4],
            "rolling_mean_4": np.mean(recent),
            "rolling_std_4": np.std(recent),
            "week": (last_date + timedelta(weeks=i + 1)).isocalendar().week,
            "month": (last_date + timedelta(weeks=i + 1)).month
        }])

        prediction = float(model.predict(prediction_features)[0])

        prediction = max(prediction, 0)

        future_date = last_date + timedelta(weeks=i + 1)

        predictions.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "predictedRate": round(prediction, 2)
        })

        rates.append(prediction)

    if len(predictions) >= 2:
        first = predictions[0]["predictedRate"]
        last = predictions[-1]["predictedRate"]

        if last > first * 1.02:
            trend = "UP"
        elif last < first * 0.98:
            trend = "DOWN"
        else:
            trend = "STABLE"
    else:
        trend = "UNKNOWN"

    return {
        "route": route,
        "forecast": predictions,
        "trend": trend
    }

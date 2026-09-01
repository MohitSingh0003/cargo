"""
app.py

CargoAI ML service — FastAPI wrapper around the trained forecast and
feasibility models. Express (cargoai-backend) calls this over HTTP via
cargoai-backend/src/services/mlClient.js and never runs the ML itself.

Run:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /                    health/status check (matches the original placeholder message)
    GET  /health              liveness + whether trained models are loaded
    POST /forecast             freight-rate forecast
    POST /feasibility          vessel/port feasibility + risk
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models.freight_forecast import forecast as run_forecast, _load_model as _load_forecast_model
from models.vessel_feasibility import check_feasibility as run_feasibility, _load_models as _load_feasibility_models
from schemas import FeasibilityRequest, FeasibilityResponse, ForecastRequest, ForecastResponse

app = FastAPI(
    title="CargoAI ML Service",
    description="Predictive freight rate forecasting and vessel/port feasibility scoring for CargoAI.",
    version="0.2.0",
)

# Express dev server default; adjust/lock down for production deployments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "CargoAI ML service is running"}


@app.get("/health")
def health():
    forecast_ready = _load_forecast_model() is not None
    feasibility_clf, risk_clf = _load_feasibility_models()
    return {
        "status": "ok",
        "models": {
            "freight_forecast": "trained" if forecast_ready else "fallback (moving average)",
            "vessel_feasibility": "trained" if feasibility_clf is not None else "fallback (rule-based)",
        },
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast_endpoint(payload: ForecastRequest):
    try:
        result = run_forecast(
            route=payload.route,
            cargo_type=payload.cargo_type,
            horizon_days=payload.horizon_days,
        )
    except Exception as exc:  # noqa: BLE001 - surface as a clean 500 for the API caller
        raise HTTPException(status_code=500, detail=f"forecast failed: {exc}") from exc
    return result


@app.post("/feasibility", response_model=FeasibilityResponse)
def feasibility_endpoint(payload: FeasibilityRequest):
    try:
        result = run_feasibility(
            vessel_type=payload.vessel_type,
            loa_m=payload.loa_m,
            beam_m=payload.beam_m,
            draft_m=payload.draft_m,
            cargo_weight_tons=payload.cargo_weight_tons,
            port=payload.port,
            port_max_loa_m=payload.port_max_loa_m,
            port_max_beam_m=payload.port_max_beam_m,
            port_max_draft_m=payload.port_max_draft_m,
            congestion_index=payload.congestion_index,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"feasibility check failed: {exc}") from exc
    return result
